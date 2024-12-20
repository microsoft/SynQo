import random

random.seed(42)
import json
from threading import Thread
import functools
import os
import re
import pandas as pd
import json
import glob
import traceback
import argparse
from utils.sql_extractor import extract_sql_query
from typing import List, Tuple
from langchain.sql_database import SQLDatabase
import os
from utils.llm_apis import getCompletionGPT4
from retry import retry
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                Exception(
                    "function [%s] timeout [%s seconds] exceeded!"
                    % (func.__name__, timeout)
                )
            ]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print("error starting thread")
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


@timeout(10)  # to prevent from freezing when execution sql takes too much time
def execute_query(query, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def get_database_schema(DB_URI: str) -> str:
    """Get the database schema from the database URI

    Args:
        DB_URI (str): Database URI

    Returns:
        str: Database schema
    """
    db = SQLDatabase.from_uri("sqlite:///" + DB_URI)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw()


def extract_schema_links(input_text: str) -> List[str]:
    pattern = r"Schema_links:\s*\[(.*?)\]"
    match = re.search(pattern, input_text)
    if match:
        schema_links_str = match.group(1)
        schema_links = [link.strip() for link in schema_links_str.split(",")]
        return schema_links
    else:
        return []


def extract_label_and_sub_questions(input_text: str) -> Tuple[str, List[str]]:
    label_pattern = r"Label:\s*(.*?)$"
    sub_questions_pattern = r"sub_questions:\s*\[(.*?)\]"

    label_match = re.search(label_pattern, input_text)
    sub_questions_match = re.search(sub_questions_pattern, input_text)

    label = label_match.group(1) if label_match else None
    label = label.replace('"', "").replace("'", "")

    sub_questions = []
    if sub_questions_match:
        sub_questions_str = sub_questions_match.group(1)
        sub_questions = [question.strip() for question in sub_questions_str.split(",")]
    return label, sub_questions


def update_json_file(json_filename, index, sql_query, db_id):
    try:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    data[str(index)] = f"{sql_query}\t----- bird -----\t{db_id}"

    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def table_descriptions_parser(database_dir):
    csv_files = glob.glob(f"{database_dir}/*.csv")
    # Iterate over the CSV files
    db_descriptions = ""
    for file_path in csv_files:
        table_name: str = os.path.basename(file_path).replace(".csv", "")
        db_descriptions += f"Table: {table_name}\n"
        table_df = pd.read_csv(file_path, encoding="latin-1")
        for _, row in table_df.iterrows():
            try:
                if pd.notna(row[2]):
                    col_description = re.sub(r"\s+", " ", str(row[2]))  # noqa: E501
                    val_description = re.sub(r"\s+", " ", str(row[4]))
                    if pd.notna(row[4]):
                        db_descriptions += f"Column {row[0]}: column description -> {col_description}, value description -> {val_description}\n"  # noqa: E501
                    else:
                        db_descriptions += f"Column {row[0]}: column description -> {col_description}\n"  # noqa: E501
            except Exception as e:
                print(e)
                db_descriptions += "No column description"
        db_descriptions += "\n"
    return db_descriptions


# Function to call GPT-4 API and log the interactions
def call_gpt4(
    prompt,
    model_key_name,
    data_params,
    prompt_type,
    trajectory_log,
    caller_agent="unknown",
):
    if prompt_type == "string":
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    response = getCompletionGPT4(
        messages, model_name=model_key_name, data_params=data_params, retry=False
    )

    # Log the interaction
    trajectory_log.append(
        {"prompt": prompt, "response": response, "caller_agent": caller_agent}
    )

    return response


@timeout(30)  # to prevent from freezing when execution sql takes too much time
def compare_sql_results(predicted_sql, ground_truth, db_path):
    predicted_sql = extract_sql_query(predicted_sql, return_None=False)
    ground_truth = extract_sql_query(ground_truth, return_None=False)
    try:
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = False

        if set(predicted_res) == set(ground_truth_res):
            res = True

        cursor.close()
        cursor.connection.close()
        return res
    except Exception as e:
        return False


@retry((Exception), tries=30, delay=50, backoff=0)
def do_it_per_thread(
    index, dev_db_path, row, timestr, without_self_correction_pred_dicts, lock
):
    db_uri = dev_db_path + "/" + row["db_id"] + "/" + row["db_id"] + ".sqlite"
    db_descriptions = (
        dev_db_path + "/" + row["db_id"] + "/" + "database_description"
    )  # noqa: E501
    columns_descriptions = table_descriptions_parser(db_descriptions)
    schema = get_database_schema(db_uri)
    question = row["question"]
    hint = row["evidence"]
    question_id = row["question_id"]
    sql_query = without_self_correction_pred_dicts[str(question_id)].split(
        "----- bird -----"
    )[0]

    SYSTEM_SELF_CORRECTION_PROMPT = f"""Database Schema:
    {schema}
    {columns_descriptions}
    """  # noqa: E501
    HUMAN_SELF_CORRECTION_PROMPT = f"""
# SQL Query Correction Guidelines
This document serves as a guideline for correcting SQL queries based on specific feedback. It aims to help in identifying common mistakes and providing a structured approach to rectify them, ensuring the queries accurately fulfill the requirements.

## Guideline Format:

[number]. **[Reminder of Mistake]**
- Question: \"Question\"
- **Incorrect SQL generated by me**: ```Incorrect SQL```
- **Corrected SQL generated by me**: ```sql Corrected SQL```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 

## Recent Mistakes and Corrections:

1. **Limit Clause Omission**
- Question: \"Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.\"
- **Incorrect SQL generated by me**: ```SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC;```
- **Corrected SQL generated by me**: ```sql SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1;```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I consider if the question specifies a limit on the number of results?
    - Have I checked if the results need to be restricted to meet the question's requirements?

2. **Incorrect Filtering Condition**
- Question: \"Find the professor ID and position in faculty who taught high-level undergraduate course of less than 10 in ID.\"
- **Incorrect SQL generated by me**: ```SELECT person.p_id, person.hasPosition FROM person INNER JOIN taughtBy ON person.p_id = taughtBy.p_id INNER JOIN course ON taughtBy.course_id = course.course_id WHERE course.courseLevel = 'Level_400' AND course.course_id < 10 AND person.professor = 0```
- **Corrected SQL generated by me**: ```sql SELECT p.p_id, p.hasPosition FROM person p INNER JOIN taughtBy tb ON p.p_id = tb.p_id INNER JOIN course c ON tb.course_id = c.course_id WHERE c.courseLevel = 'Level_400' AND c.course_id < 10 AND p.professor = 1```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Have I accurately identified and applied the correct filtering conditions based on the question's requirements?
    - Did I verify the logical operators and conditions to ensure they align with the intended query logic?

3. **Misinterpretation of Requirements**
- Question: \"Among the faculty affiliated professor, how many professors teaches professional or master/undergraduate courses?\"
- **Incorrect SQL generated by me**: ```SELECT COUNT(DISTINCT T1.p_id) FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.professor = 0 AND T1.hasPosition = 'Faculty' AND T3.courseLevel = 'Level_500'```
- **Corrected SQL generated by me**: ```sql SELECT COUNT(DISTINCT T1.p_id) FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.professor = 1 AND T1.hasPosition = 'Faculty_aff' AND T3.courseLevel = 'Level_500'```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I fully understand the question's requirements before writing the query?
    - Have I ensured that all conditions and filters accurately reflect the question's intent?

4. **Incorrect Aggregation and Calculation**
- Question: \"What is the average number of professional or master/undergraduate courses being taught by each professor?\"
- **Incorrect SQL generated by me**: ```SELECT COUNT(DISTINCT taughtBy.course_id) / COUNT(DISTINCT taughtBy.p_id) AS average_courses_per_professor FROM taughtBy INNER JOIN course ON taughtBy.course_id = course.course_id WHERE course.courseLevel = 'Level_500'```
- **Corrected SQL generated by me**: ```sql SELECT AVG(course_count) AS average_courses_per_professor FROM (SELECT COUNT(*) AS course_count FROM taughtBy INNER JOIN course ON taughtBy.course_id = course.course_id WHERE course.courseLevel = 'Level_500' GROUP BY taughtBy.p_id) AS subquery```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Have I used the correct aggregation functions to calculate averages or totals as required by the question?
    - Did I consider using subqueries for complex calculations to ensure accuracy?

5. **Counting Specific Column vs. All Rows**
- Question: \"How many male users are in the age group of M32-38?\"
- **Incorrect SQL generated by me**: ```SELECT COUNT(*) FROM gender_age WHERE gender = 'M' AND group = 'M32-38'```
- **Corrected SQL generated by me**: ```sql SELECT COUNT(gender) FROM gender_age WHERE gender = 'M' AND `group` = 'M32-38'```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I use `COUNT(*)` when I should have specified the column I was interested in counting?
    - Have I ensured to use backticks around reserved keywords when they are used as column names?

6. **Directly Targeting the Youngest Age**
- Question: \"What is the gender of the youngest user?\"
- **Incorrect SQL generated by me**: ```SELECT gender FROM gender_age ORDER BY age ASC LIMIT 1```
- **Corrected SQL generated by me**: ```sql SELECT gender FROM gender_age WHERE age = (SELECT MIN(age) FROM gender_age)```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I consider the most efficient way to directly target the desired value?
    - Have I evaluated if using a subquery could provide a more accurate and efficient solution?

7. **Ordering and Limiting for Maximum Value Retrieval**
- Question: \"What is the age of the oldest active user that participated in the event held on 5/6/2016 at coordinates 121, 31?\"
- **Incorrect SQL generated by me**: ```SELECT MAX(gender_age.age) FROM gender_age INNER JOIN events ON gender_age.device_id = events.device_id INNER JOIN app_events ON events.event_id = app_events.event_id WHERE app_events.is_active = 1 AND events.timestamp LIKE '2016-05-06%' AND events.longitude = 121 AND events.latitude = 31```
- **Corrected SQL generated by me**: ```sql SELECT gender_age.age FROM gender_age INNER JOIN events_relevant AS er ON gender_age.device_id = er.device_id INNER JOIN app_events ON er.event_id = app_events.event_id WHERE app_events.is_active = 1 AND SUBSTR(er.timestamp, 1, 10) = '2016-05-06' AND er.longitude = 121 AND er.latitude = 31 ORDER BY gender_age.age DESC LIMIT 1```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I correctly use ordering and limiting to retrieve the maximum or minimum value?
    - Have I ensured the conditions and joins are accurately targeting the required data?

8. **Manual Calculation of Average**
- Question: \"What is the average score of the movie 'The Fall of Berlin' in 2019?\"
- **Incorrect SQL generated by me**: ```SELECT AVG(rating_score) FROM ratings INNER JOIN movies ON ratings.movie_id = movies.movie_id WHERE movies.movie_title = 'The Fall of Berlin' AND rating_timestamp_utc LIKE '2019%'```
- **Corrected SQL generated by me**: ```sql SELECT CASE WHEN COUNT(r.rating_id) = 0 THEN NULL ELSE SUM(r.rating_score) / COUNT(r.rating_id) END AS average_score FROM ratings AS r INNER JOIN movies AS m ON r.movie_id = m.movie_id WHERE m.movie_title = 'The Fall of Berlin' AND r.rating_timestamp_utc >= '2019-01-01' AND r.rating_timestamp_utc < '2020-01-01'```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Have I considered performing manual calculations for more control over the result?
    - Did I use precise date filtering methods to ensure accuracy?

9. **Simplifying Date Filtering**
- Question: \"Indicate the location of all the events that occurred on April 30, 2016.\"
- **Incorrect SQL generated by me**: ```SELECT * FROM table WHERE timestamp BETWEEN '2016-04-30 00:00:00' AND '2016-04-30 23:59:59'```
- **Corrected SQL generated by me**: ```sql SELECT longitude, latitude FROM events WHERE date(timestamp) = '2016-04-30'```
- **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I consider the simplest and most effective method for date filtering?
    - Have I avoided unnecessary complexity in filtering by date and time?

10. **Incorrect Table and Condition Use for App Installation Analysis**
    - Question: \"On which brand of phone are the most applications installed?\"
    - **Incorrect SQL generated by me**: ```SELECT T1.phone_brand, COUNT(*) AS installed_count FROM phone_brand_device_model2 AS T1 JOIN events AS T2 ON T1.device_id = T2.device_id JOIN app_events AS T3 ON T2.event_id = T3.event_id WHERE T3.is_installed = 1 GROUP BY T1.phone_brand ORDER BY installed_count DESC LIMIT 1```
    - **Corrected SQL generated by me**: ```sql SELECT T1.phone_brand, COUNT(*) AS active_count FROM phone_brand_device_model2 AS T1 JOIN events_relevant AS T2 ON T1.device_id = T2.device_id JOIN app_events_relevant AS T3 ON T2.event_id = T3.event_id WHERE T3.is_active = 1 GROUP BY T1.phone_brand ORDER BY active_count DESC LIMIT 1```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I ensure to use the correct tables that contain the relevant data for my analysis?
    - Have I correctly identified the condition that matches the question's intent (active vs. installed)?

11. **Misuse of DISTINCT in Counting Unique Device IDs**
    - Question: \"How many men under the age of 23 have apps installed but are not active on their devices?\"
    - **Incorrect SQL generated by me**: ```SELECT COUNT(DISTINCT gender_age.device_id) FROM gender_age INNER JOIN events ON gender_age.device_id = events.device_id INNER JOIN app_events ON events.event_id = app_events.event_id WHERE gender_age.gender = 'M' AND gender_age.age < 23 AND app_events.is_installed = 1 AND app_events.is_active = 0```
    - **Corrected SQL generated by me**: ```sql SELECT COUNT(gender_age.device_id) FROM gender_age INNER JOIN events_relevant ON gender_age.device_id = events_relevant.device_id INNER JOIN app_events_relevant ON events_relevant.event_id = app_events_relevant.event_id WHERE gender_age.gender = 'M' AND gender_age.age < 23 AND app_events_relevant.is_active = 0```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I unnecessarily use `DISTINCT` when the query logic or data model does not require it?
    - Have I ensured that my joins and conditions accurately reflect the data's structure and the question's intent?

12. **Date Filtering and Table Naming for Event Analysis**
    - Question: \"Which gender logged in the most to an event in the first 10 days of May 2016?\"
    - **Incorrect SQL generated by me**: ```SELECT gender, COUNT(*) AS login_count FROM gender_age INNER JOIN events ON gender_age.device_id = events.device_id WHERE timestamp BETWEEN '2016-05-01 00:00:00' AND '2016-05-10 23:59:59' GROUP BY gender ORDER BY login_count DESC LIMIT 1```
    - **Corrected SQL generated by me**: ```sql SELECT T.gender, COUNT(T.device_id) AS login_count FROM (SELECT gender_age.gender, gender_age.device_id FROM gender_age INNER JOIN events_relevant ON gender_age.device_id = events_relevant.device_id WHERE date(events_relevant.timestamp) BETWEEN '2016-05-01' AND '2016-05-10') AS T GROUP BY T.gender ORDER BY login_count DESC LIMIT 1```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I use the most accurate method for date filtering to match the question's requirements?
    - Have I selected the correct tables and used aliases for clarity and efficiency in my query?

13. **Accuracy in Calculating Average Age for Specific Conditions**
    - Question: \"Calculate the average age of people who have apps installed but are not active on their devices.\"
    - **Incorrect SQL generated by me**: ```SELECT AVG(ga.age) AS average_age FROM gender_age ga JOIN events e ON ga.device_id = e.device_id JOIN app_events ae ON e.event_id = ae.event_id WHERE ae.is_installed = 1 AND ae.is_active = 0;```
    - **Corrected SQL generated by me**: ```sql SELECT AVG(gender_age.age) FROM gender_age JOIN events_relevant ON gender_age.device_id = events_relevant.device_id JOIN app_events_relevant ON events_relevant.event_id = app_events_relevant.event_id WHERE app_events_relevant.is_installed = 1 AND app_events_relevant.is_active = 0```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Have I ensured to use the correct tables for a more accurate analysis?
    - Did I correctly apply conditions to match the specific scenario described in the question?

14. **Selecting Specific Columns for Efficiency**
    - Question: \"Please list any three events that happened on the 1st of May 2016 that have the same latitude of 31.\"
    - **Incorrect SQL generated by me**: ```SELECT * FROM events WHERE timestamp LIKE '2016-05-01%' AND latitude = 31 LIMIT 3```
    - **Corrected SQL generated by me**: ```sql SELECT event_id FROM events WHERE timestamp LIKE '2016-05-01%' AND latitude = 31 LIMIT 3```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I only select the columns necessary for the question's requirements, ensuring efficiency?
    - Have I used the correct filtering criteria to accurately target the desired data?

15. **Correcting Device ID and Aggregation Method**
    - Question: \"What is the difference between the events of device number -9222956879900150000 that can be located and those that are unable to be located?\"
    - **Incorrect SQL generated by me**: ```SELECT (SUM(CASE WHEN latitude != 0 AND longitude != 0 THEN 1 ELSE 0 END) - SUM(CASE WHEN latitude = 0 AND longitude = 0 THEN 1 ELSE 0 END)) AS location_difference FROM events WHERE device_id = -9222956879900150000;```
    - **Corrected SQL generated by me**: ```sql SELECT SUM(IIF(latitude != 0 AND longitude != 0, 1, 0)) - SUM(IIF(latitude = 0 AND longitude = 0, 1, 0)) AS difference FROM events WHERE device_id = '-922956879900150000'```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I verify the accuracy of key identifiers such as device IDs before executing the query?
    - Have I utilized the most efficient aggregation method to achieve the desired calculation?

16. **Table Naming and Alias Usage for Clarity**
    - Question: \"Show the avatar of the user who gave the rating at 2019/10/17 1:36:36.\"
    - **Incorrect SQL generated by me**: ```SELECT ratings_users.user_avatar_image_url FROM ratings INNER JOIN ratings_users ON ratings.user_id = ratings_users.user_id WHERE ratings.rating_timestamp_utc = '2019-10-17 01:36:36'```
    - **Corrected SQL generated by me**: ```sql SELECT lists_users.user_avatar_image_url FROM ratings INNER JOIN lists_users ON ratings.user_id = lists_users.user_id WHERE ratings.rating_timestamp_utc = '2019-10-17 01:36:36'```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Have I ensured to reference the correct tables as per the question's context?
    - Did I use aliases where appropriate to enhance the readability and clarity of my query?

17. **Direct Counting Without Unnecessary Distinct**
    - Question: \"How many users belong to the same behavior category as comics?\"
    - **Incorrect SQL generated by me**: ```SELECT COUNT(DISTINCT T1.app_id) FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id WHERE T2.category = 'comics'```
    - **Corrected SQL generated by me**: ```sql SELECT COUNT(app_id) FROM app_labels INNER JOIN label_categories ON app_labels.label_id = label_categories.label_id WHERE category = 'comics'```
    - **Negative and strict step-by-step ask-to-myself questions to prevent the same mistake again**: 
    - Did I unnecessarily use `DISTINCT` when the query logic does not require it?
    - Have I ensured that my conditions accurately target the required data without adding unnecessary complexity?


### Question:  {question}
### Hint:  {hint}
### Initial sql query:  ```sql {sql_query}```
### Asking myself: step-by-step reasoning

    """.format()
    prompt = [
        {
            "role": "system",
            "content": SYSTEM_SELF_CORRECTION_PROMPT,
        },
        {
            "role": "user",
            "content": HUMAN_SELF_CORRECTION_PROMPT,
        },
    ]
    model_key_name = "gpt-4-turbo-0125-spot"
    data_params = {
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "n": 1,
        "temperature": 0.0,
    }
    # print("correction_prompt: ", prompt)
    correction = call_gpt4(
        prompt,
        model_key_name,
        data_params,
        prompt_type="message",
        trajectory_log=[],
        caller_agent="unknown",
    )

    finall_sql = extract_sql_query(correction, return_None=False)
    if finall_sql is not None:
        one_liner_sql_query = finall_sql.replace("\n", "").replace("\r", "")
    else:
        if sql_query is not None:
            one_liner_sql_query = sql_query.replace("\n", "").replace("\r", "")
        else:
            one_liner_sql_query = "SELECT * FROM table"  # no query generated, placeholder to avoid errors # noqa: E501
    to_print_str = f""""
        --------------------------------------------------
        Processing row: " {index}
        "Question: " {question}
        "Hint: " {hint}
        "Database: " {db_uri}
        initial sql query: {without_self_correction_pred_dicts[str(question_id)].split(
                "----- bird -----"
            )[0]}
        Gold sql query: {row["SQL"]}
        self-corrected final sql query: {one_liner_sql_query}
        "correction output: ", {correction}
        """
    with lock:
        update_json_file(
            "./results/dev_self_correction_with_MAGIC_bird.json".format(
                model_key_name, timestr
            ),
            index,
            one_liner_sql_query,
            row["db_id"],
        )
        with open(
            "./results/dev_self_correction_with_MAGIC_bird.log".format(
                model_key_name, timestr
            ),
            "a",
        ) as fw:

            fw.write(to_print_str)
    print(to_print_str)


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Provide arguments")

    # Add the arguments
    parser.add_argument(
        "--model_key_name",
        type=str,
        default="gpt-4o",
        help="The model key name (default: gpt-4o)",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=5,
        help="The maximum number of threads (default: 5)",
    )
    parser.add_argument(
        "--initial_pred_path",
        type=str,
        default="",
        help="Initial systems preds path",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Databases path where the tables are stored",
    )

    parser.add_argument(
        "--input_df_path",
        type=str,
        required=True,
        help="Input dataframe file path",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    model_key_name = args.model_key_name
    max_threads = args.max_threads
    initial_pred_path = args.initial_pred_path
    without_self_correction_pred_dicts = json.loads(open(initial_pred_path, "r").read())

    api_key, api_version, azure_endpoint, model = "", "", "", "" # set accordingly

    # CHANGE THIS TO YOUR OPENAI API KEY
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
    dev_db_path = args.db_path
    dev_df = pd.read_json(args.input_df_path)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(
                do_it_per_thread,
                index,
                dev_db_path,
                row,
                without_self_correction_pred_dicts,
                lock,
            )
            for index, row in dev_df.iterrows()
        ]

        # Wait for all futures to complete
        for future in futures:
            # Optional: Handle the result or exception if needed
            try:
                future.result()
            except Exception as e:
                print(
                    "skip due to the following error: {}".format(traceback.format_exc())
                )
                pass
    out_path = "./results/dev_self_correction_with_MAGIC_bird.json"
    outjson = json.loads(open(out_path, "r").read())

    # sorting results as we did multi-thread generation we need to sort results here.
    for check_existing in range(0, len(dev_df) - 1):
        if str(check_existing) not in outjson and int(check_existing) not in outjson:
            outjson[str(check_existing)] = ""
    outjson = {
        str(k): v
        for k, v in sorted(
            outjson.items(),
            key=lambda item: int(item[0]),
        )
    }
    with open(out_path, "w") as fp:
        json.dump(out_path, fp=fp, indent=True)

    # replacing failure of self-correction with initial prediction
    for key, sql in outjson.items():
        db_id = sql.split("\t----- bird -----\t")[1]
        db_uri = dev_db_path + "/" + db_id + "/" + db_id + ".sqlite"
        if "SELECT * FROM table" in sql:
            outjson[key] = without_self_correction_pred_dicts[key]
        try:
            execute_query(sql.split("\t----- bird -----\t")[0], db_uri)
        except Exception as e:
            outjson[key] = without_self_correction_pred_dicts[key]
    with open(out_path, "w") as fp:
        json.dump(outjson, fp=fp, indent=True)
    #####################
    print("All API calls had been done!.")
