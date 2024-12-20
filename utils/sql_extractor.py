import re


def extract_sql_query(
    input_text, return_None=True, output_of_gpt=True, to_return_val=None
):
    try:
        input_text = input_text.replace("YOU MUST ONLY RETURN SQL.", " ").strip()
        # Use regular expression to find all SQL queries between sql\n and ```
        cleaned_input_text = (
            input_text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")
        )
        matches = re.findall(r"```sql(.*?)```", cleaned_input_text, re.DOTALL)
        if matches:
            latest_sql_query = matches[-1]
            latest_sql_query = latest_sql_query.replace("\\n", " ").strip()
            return latest_sql_query
        else:
            if return_None == True:
                return None
            else:
                input_text = (
                    input_text.split("SQL: ")[-1:][0]
                    .split("Revised_SQL: ")[-1:][0]
                    .strip()
                )
                # if "SELECT" in input_text:
                #     input_text = "SELECT " + input_text.split("SELECT")[1:][0]
                if to_return_val is None:
                    return input_text
                else:
                    return to_return_val
    except Exception as e:
        if return_None:
            print("error happened, none returned. Error: {}".format(e))
            return None
        else:
            print("error happened, input_text returned. Error: {}".format(e))
            return input_text if to_return_val is None else to_return_val

