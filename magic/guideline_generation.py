import pandas as pd
import argparse
import json
import glob
import time
import jsonlines
from utils.sql_extractor import extract_sql_query
from utils.llm_apis import getCompletionGPT4


def call_gpt4(
    prompt,
    model_key_name,
    data_params,
    prompt_type,
    trajectory_log,
    caller_agent="unknown",
):
    if prompt_type == "string":
        messages = [{"role": "user", "content": prompt}]
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


def main(initial_pred_path, gold_df_path, trajectory_path, guideline_out_path):
    # Load prediction object
    with open(initial_pred_path, "r") as f:
        pred_obj = json.load(f)

    # Load gold dataframe
    gold_df = pd.read_json(gold_df_path)

    # Get the current time string
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Format the guideline output trajectory_path
    guideline_out_path = guideline_out_path.format(timestr=timestr)

    # Initialize an empty guideline
    guideline_format = """
    [number]. **[Reminder of mistake]**
       - Question: "Question"
       - **Incorrect SQL generated by me**: ```Incorrect corrected sql ```
       - **Corrected SQL generated by me**: ```sql corrected sql ```
       - **Negative and strict step-by-step ask-to-myself questions to prevent same mistake again**: 
    """
    current_guideline = """
    """

    files = list(glob.glob(trajectory_path))
    model_key_name = "gpt-4"
    data_params = {
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "n": 1,
        "temperature": 0.0,
    }

    guideline_materials = []

    for index_, f in enumerate(files):
        try:
            file_name = f.split("\\")[-1]
            question_id = file_name.split("trajectory-")[1].split("-success")[0]
            succsess = "True" in file_name
            if succsess:
                question = gold_df.iloc[int(question_id)]["question"]
                if "ratio" in question:
                    continue
                trajectory_object = json.loads(open(f, "r").read())
                latest_correct_sql = trajectory_object[-1]["response"]
                latest_correct_sql = extract_sql_query(
                    latest_correct_sql, return_None=False
                )

                latest_feedback_that_worked = str(trajectory_object[-2]["response"])
                initially_incorrect_predicted_sql = pred_obj[str(question_id)]

                all_incorrect_sqls_by_correction_agents = []
                for obj in trajectory_object[0:-1]:
                    if (
                        "caller_agent" in obj
                        and "correction_agent_call" in obj["caller_agent"]
                    ):
                        failed_correction = extract_sql_query(
                            obj["response"], return_None=False
                        )
                        all_incorrect_sqls_by_correction_agents.append(
                            failed_correction
                        )

                guideline_material = f"""
    Question: {question}
    Feedback: {latest_feedback_that_worked}
    Incorrect sql 1: {initially_incorrect_predicted_sql}
                """
                for incorrect_sql_index, correction_sql in enumerate(
                    all_incorrect_sqls_by_correction_agents
                ):
                    guideline_material += f"""
    Incorrect sql {incorrect_sql_index + 2}: {correction_sql},
    """
                guideline_material += f"""
    Successfully Corrected SQL using the feedback: {latest_correct_sql}
                """
                guideline_materials.append(guideline_material)

                if len(guideline_materials) >= 10:
                    print("Updating guideline....")
                    user_prompt = f"""
    # Guideline format:
    {guideline_format}

    # Guideline so far:
    {current_guideline}

    # Recent mistakes that must be aggregate to Guideline:
    {guideline_materials}

    # Updated Guideline (Return the entire guideline):
    """

                    prompt = [{"role": "user", "content": user_prompt}]
                    prompt_type = "message"
                    no_error = False
                    while not no_error:
                        try:
                            current_guideline = call_gpt4(
                                prompt,
                                model_key_name,
                                data_params,
                                prompt_type,
                                [],
                                caller_agent="unknown",
                            )
                            no_error = True
                        except Exception as e:
                            print(f"Retrying... error was: {e}")
                            pass

                    with jsonlines.open(guideline_out_path, mode="a") as jsonl_write:
                        obj = {f"guideline_iteration_{index_}": current_guideline}
                        jsonl_write.write(obj)
                    guideline_materials = []
                    print("Updated!")
        except Exception as e:
            pass
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating guideline by MAGIC.")
    parser.add_argument(
        "--initial_pred_path",
        type=str,
        default=r"./data/bird/train_initial_pred.json",
        required=True,
        help="Path to the initial prediction system file.",
    )

    parser.add_argument(
        "--gold_df_path",
        type=str,
        default=r"./data/bird/train_df.json",
        required=True,
        help="Path to the gold data frame file.",
    )

    parser.add_argument(
        "--trajectory_path",
        type=str,
        default=r"./src/results/MAGIC-trajectory/*",
        required=True,
        help="Path to the directory or files to process.",
    )

    parser.add_argument(
        "--guideline_out_path",
        type=str,
        default=r"./src/results/MAGIC-Guideline/guideline_progress_per_batch.json",
        required=True,
        help="Template trajectory_path for the guideline output file.",
    )

    args = parser.parse_args()

    initial_pred_path = args.initial_pred_path
    gold_df_path = args.gold_df_path
    trajectory_path = args.trajectory_path
    guideline_out_path = args.guideline_out_path

    print("Initial Prediction Path:", initial_pred_path)
    print("Gold Data Frame Path:", gold_df_path)
    print("Trajectory Path:", trajectory_path)
    print("Guideline Output Path Template:", guideline_out_path)

    main(initial_pred_path, gold_df_path, trajectory_path, guideline_out_path)
    # python3 -u -m --initial_pred_path "./data/bird/train_initial_pred.json" --gold_df_path "./data/bird/train_df.json"  --trajectory_path "./src/results/MAGIC-trajectory/*" --guideline_out_path "./src/results/MAGIC-Guideline/guideline_progress_per_batch.json"
