import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import traceback
import pandas as pd
from utils.llm_apis import getCompletionGPT4
import glob
import sqlite3
import re
import json
from langchain.sql_database import SQLDatabase
import os
from utils.sql_extractor import extract_sql_query
import re
from threading import Thread
import functools


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
        print(
            f"One of sqls could not be executed. We skipp as false match in this situation"  #! predicted_sql: {predicted_sql} and ground_truth is : {ground_truth}"
        )
        return False


# Function to call GPT-4 API and log the interactions
def call_gpt4(
    prompt,
    model_key_name,
    data_params,
    prompt_type,
    trajectory_log,
    caller_agent,
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


# Function to generate feedback using Feedback LLM
def feedback_agent(
    prompt,
    model_key_name,
    data_params,
    trajectory_log,
    caller_agent,
    evidence,
    iteration,
    prompt_type,
    manager_additional_command="",
):
    print("feedback_agent -- iteration:", iteration)

    try:
        feedback = call_gpt4(
            prompt,
            model_key_name,
            data_params,
            prompt_type=prompt_type,
            trajectory_log=trajectory_log,
            caller_agent=caller_agent,
        )
    except Exception as e:
        feedback = call_gpt4(
            prompt,
            model_key_name,
            data_params,
            prompt_type="string",
            trajectory_log=trajectory_log,
            caller_agent=caller_agent,
        )
        print("error happened in first try so we try as string")
        pass

    print(
        "\n\n---------------------------------------------------$$$$$$$$$$$$$----------------------------------------------\n"
    )
    # print("feedback that is returned: ", feedback)
    return feedback


def get_initial_prompt_for_correction_agent(
    nl_question,
    schema,
    columns_descriptions,
    incorrect_sql,
    feedback,
):
    print("call func: get_initial_prompt_for_correction_agent")
    system_prompt = f"""
Your task is to correct the predicted SQL based on the provided feedback by expert human.

##
Example of Asking myself: step-by-step reasoning
SELECT COUNT(*) FROM major WHERE college = "College of Humanities and Social Sciences"
1. **Did I use the correct table for the query?**
   - Yes, the `major` table contains the `college` column which is necessary for filtering the majors based on the college name.

2. **Did I correctly specify the column to count?**
   - Yes, using `COUNT(*)` is appropriate here since we are interested in the total number of majors in the specified college, not a specific column.

3. **Did I use the correct filtering condition?**
   - I need to ensure that the filtering condition accurately matches the college name as specified in the question. The use of double quotes for string literals in SQL might be incorrect depending on the SQL dialect. Some SQL dialects prefer single quotes for string literals.

4. **Did I unnecessarily use `DISTINCT`?**
   - No, `DISTINCT` is not used in the initial query, which is correct because we want to count all majors, not just unique ones.

5. **Have I ensured that my conditions accurately target the required data without adding unnecessary complexity?**
   - The condition seems straightforward and targets the required data accurately by filtering majors based on the college name.

6. **Did I use the correct syntax for string literals?**
   - The initial query used double quotes for the string literal, which might not be correct for all SQL dialects. It's safer to use single quotes for string literals.

Revised SQL:
```sql
SELECT COUNT(*) FROM major WHERE college = 'College of Humanities and Social Sciences'
```

##

1. Input Information: You will receive a question, a database schema, a predicted SQL query, and a human feedback.

2. SQL format in your response:
    - You must ensure that your response contains a valid SQL which is either corrected SQL or the predicted SQL without any change if you think it is correct already.
    - The format of SQL in your response must be in the following format: ```sql\n SQL \n```. Example of format:  ```sql\n SELECT * from users \n```
    """

    user_prompt = f"""
- Schema Overview:  
{schema}

- Column Details:  
{columns_descriptions}

####  
- Question: {nl_question}  

- Predicted SQL:   
```sql  
{incorrect_sql}
```

- Expert Human Feedback: {feedback}

- Asking myself: step-by-step reasoning
"""
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return prompt


def correction_agent(
    prompt,
    data_params,
    model_key_name,
    trajectory_log,
    caller_agent,
    iteration,
    prompt_type,
):
    print("correction agent --- {}".format(caller_agent))
    try:
        # print("correction agent - inside try --- {}".format(caller_agent))
        correction_sql = call_gpt4(
            prompt=prompt,
            model_key_name=model_key_name,
            data_params=data_params,
            prompt_type=prompt_type,
            trajectory_log=trajectory_log,
            caller_agent=caller_agent,
        )
    except Exception as e:
        # print("correction except - inside try --- {}".format(caller_agent))

        print("error happened in first try so we try as string")
        correction_sql = call_gpt4(
            prompt=prompt,
            model_key_name=model_key_name,
            data_params=data_params,
            prompt_type="string",
            trajectory_log=trajectory_log,
            caller_agent=caller_agent,
        )
        pass

    # print("correction except - before return --- {}".format(caller_agent))

    print(
        "\n\n---------------------------------------------------$$$$$$$$$$$$$----------------------------------------------\n"
    )
    # print("correction-agent return: ", correction_sql)
    return correction_sql


def get_initial_prompt_for_feedback_agent(
    nl_question,
    predicted_sql_at_first,
    ground_truth_sql,
    correction_sqls,
    evidence,
):
    print("call func: get_initial_prompt_for_feedback_agent")
    system_prompt = """Complete the text in chat style like a database manager expert. Write in simple present without using correct SQL.  Accept what the user identifies as correct or incorrect."""
    user_prompt = f"""
"question": "{nl_question}",
"evidence": "{evidence}",
"Correct SQL": "{ground_truth_sql}",
"Incorrect sql": "{predicted_sql_at_first}",

"""
    for correction_sql in correction_sqls:
        user_prompt += f"""
"Incorrect sql": "{correction_sql}",

"""

    user_prompt += f"""Incorrect SQLs mistakes are:"""
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return prompt


def manager_revises_prompt(
    agent_prompts,
    agent_outputs,
    agent_type,
    model_key_name,
    data_params,
    iteration_of_try,
    trajectory_log,
    ideal_output_of_assistant_agent=None,
):
    print(
        "manager_revises_prompt: ",
    )
    my_data_params = dict(data_params)
    my_data_params["max_tokens"] = 2000
    if agent_type == "correction":
        agent_description = "This agent generates corrections for SQL queries based on expert human feedback."
    elif agent_type == "feedback":
        agent_description = "This agent generates feedback based on the comparison between predicted and ground truth SQL queries."

    system_prompt = """
You are a helpful AI assistant that manages other assistants.
"""
    user_prompt = f"""
Manager, please review the following prompt for the following agent and generate a revised prompt. 
So far you have revised the prompt for {iteration_of_try} times.
Agent description: {agent_description}
Previoust output of agent that was not useful: {agent_outputs[-1]}
Previous prompt of agent that you should revise: {agent_prompts[-1]}
"""
    if ideal_output_of_assistant_agent is not None:
        user_prompt += f"""
    #### 
    
    The ideal output of agent should be the following but we cannot directly give the ideal output to the agent:
    ideal output of agent: {ideal_output_of_assistant_agent}
    """

    user_prompt += """

####

Revise prompt (Return the entire prompt so it can be directly pass to the agent):

"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    revised_prompt = call_gpt4(
        prompt,
        model_key_name,
        my_data_params,
        prompt_type="message",
        trajectory_log=trajectory_log,
        caller_agent="manager_revises_prompt_itteration_{}".format(iteration_of_try),
    )
    print(
        "\n\n---------------------------------------------------$$$$$$$$$$$$$----------------------------------------------\n"
    )
    # print("return revised_prompt: ", revised_prompt)
    try:
        json_object = json.loads(revised_prompt)
        return (json_object, "message")
    except Exception as e:
        return (revised_prompt, "string")
        pass


def manager_agent(
    question_id,
    index,
    nl_question,
    evidence,
    incorrect_sql,
    ground_truth_sql,
    schema,
    columns_descriptions,
    db_path,
    model_key_name,
    data_params,
    max_try,
    trajectory_path,
):
    print(f"Question id started: {index}\n\n")
    trajectory_log = []
    try_cnt = 0
    iteration = 0
    feedback_history = []
    correction_history = []
    correction_sqls = []
    feedback_agent_prompts = []
    correction_agent_prompts = []
    while try_cnt == 0 or (
        compare_sql_results(correction_sql, ground_truth_sql, db_path) == False
        and try_cnt <= max_try
    ):
        try_cnt += 1
        iteration += 1
        correction_revise_prompt_type = ""
        feedback_revise_prompt_type = ""

        if iteration > 1:
            feedback_prompt, feedback_revise_prompt_type = manager_revises_prompt(
                agent_prompts=feedback_agent_prompts,
                agent_outputs=feedback_history,
                agent_type="feedback",
                model_key_name=model_key_name,
                data_params=data_params,
                iteration_of_try=iteration,
                trajectory_log=trajectory_log,
            )
        else:
            feedback_revise_prompt_type = "message"
            feedback_prompt = get_initial_prompt_for_feedback_agent(
                nl_question=nl_question,
                predicted_sql_at_first=incorrect_sql,
                ground_truth_sql=ground_truth_sql,
                correction_sqls=correction_sqls,
                evidence=evidence,
            )
        feedback_agent_prompts.append(feedback_prompt)
        feedback = feedback_agent(
            prompt=feedback_prompt,
            model_key_name=model_key_name,
            data_params=data_params,
            trajectory_log=trajectory_log,
            caller_agent="feedback_agent_call_{}".format(iteration),
            evidence=evidence,
            iteration=iteration,
            prompt_type=feedback_revise_prompt_type,
            manager_additional_command="",
        )
        feedback_history.append(feedback)

        if iteration > 1:
            correction_prompt, correction_revise_prompt_type = manager_revises_prompt(
                agent_prompts=correction_agent_prompts,
                agent_outputs=correction_history,
                agent_type="correction",
                model_key_name=model_key_name,
                data_params=data_params,
                iteration_of_try=iteration,
                trajectory_log=trajectory_log,
                ideal_output_of_assistant_agent=ground_truth_sql,
            )
        else:
            correction_revise_prompt_type = "message"
            correction_prompt = get_initial_prompt_for_correction_agent(
                nl_question=nl_question,
                schema=schema,
                columns_descriptions=columns_descriptions,
                incorrect_sql=incorrect_sql,
                feedback=feedback,
            )
        correction_agent_prompts.append(correction_prompt)
        print("correction: prompt_type", correction_revise_prompt_type)
        correction_sql = correction_agent(
            prompt=correction_prompt,
            data_params=data_params,
            model_key_name=model_key_name,
            trajectory_log=trajectory_log,
            caller_agent="correction_agent_call_{}".format(iteration),
            iteration=iteration,
            prompt_type=correction_revise_prompt_type,
        )
        correction_history.append(correction_sql)
        correction_sql = extract_sql_query(correction_sql, return_None=False)
        correction_sqls.append(correction_sql)
        print(f"Iteration {iteration}, SQL:  {correction_sql}")
    success = compare_sql_results(correction_sql, ground_truth_sql, db_path)

    trajectory_log, success, num_of_tries_to_fix = (trajectory_log, success, try_cnt)
    if success == True:
        print(f"predicted sql: {predicted_sql}")
        print(f"gold sql: {ground_truth_sql}")

    with open(
        f"{trajectory_path}/trajectory-{question_id}-success-{succsess_per_i}.json",
        "a",
    ) as fw:
        json.dump(trajectory_log, fw, indent=True)
    print(f"Question id finished: {index}\n\n")
    return


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


if __name__ == "__main__":
    # Initialize trajectory log
    trajectory_log = []

    # Create the parser
    parser = argparse.ArgumentParser(description="Provide arguments")

    # Add the arguments
    parser.add_argument(
        "--model_key_name",
        type=str,
        default="gpt-4",
        help="The model key name (default: gpt-4o)",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=10,
        help="The maximum number of threads (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The value of temperature",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Databases path where the tables are stored",
    )

    parser.add_argument(
        "--gold_df_path",
        type=str,
        required=True,
        help="Gold df path",
    )

    parser.add_argument(
        "--initial_pred_path",
        type=str,
        required=True,
        help="Initial systems preds path",
    )

    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help="trajectory_path",
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        required=False,
        help="max_iterations",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    model_key_name = args.model_key_name
    max_threads = args.max_threads
    temperature = args.temperature
    trajectory_path = args.trajectory_path
    max_try = args.max_iterations

    api_key, api_version, azure_endpoint, model = "", "", "", "" # set accordingly

    # Set environment variables for API keys and endpoint
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

    gpt_data_params = {
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 600,
        "stream": False,
        "n": 1,
        "temperature": temperature,
    }

    eval_db_path = args.db_path
    eval_df = pd.read_json(args.gold_df_path)
    pred_obj_without_self_correction = json.loads(
        open(args.initial_pred_path, "r").read()
    )

    print("max_threads: ", max_threads)

    print("start....")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for index, row in eval_df.iterrows():
            try:
                question_id = index
                db_uri = (
                    eval_db_path + "/" + row["db_id"] + "/" + row["db_id"] + ".sqlite"
                )
                db_descriptions = (
                    eval_db_path + "/" + row["db_id"] + "/" + "database_description"
                )
                columns_descriptions = table_descriptions_parser(db_descriptions)
                schema = get_database_schema(db_uri)
                question = row["question"]
                if "evidence" in row:
                    hint = str(row["evidence"])
                else:
                    hint = ""
                difficulty = "unknown_difficulty"
                ground_truth_sql = row["SQL"]

                predicted_sql = pred_obj_without_self_correction[str(question_id)]
                predicted_sql = predicted_sql.split("----- bird -----")[0].strip()
                if compare_sql_results(predicted_sql, ground_truth_sql, db_uri):
                    continue

                futures.append(
                    executor.submit(
                        manager_agent,
                        question_id,
                        index,
                        question,
                        hint,
                        predicted_sql,
                        ground_truth_sql,
                        schema,
                        columns_descriptions,
                        db_uri,
                        model_key_name,
                        gpt_data_params,
                        max_try,
                        trajectory_path,
                    )
                )
            except Exception as e:
                print(f"Error in thread {index}: {e}")
                print(traceback.format_exc())
        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(
                    "skip due to the following error: {}".format(traceback.format_exc())
                )
            pass
