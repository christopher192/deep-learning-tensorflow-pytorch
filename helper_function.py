import tensorflow as tf
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def performance_metrics(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average = "weighted")
    model_results = { "accuracy": model_accuracy, "precision": model_precision, "recall": model_recall, "f1": model_f1 }

    return model_results

def compare_baseline_with_new_result(baseline_result, new_result):
    for key, value in baseline_result.items():
        print(f"Baseline {key}: {value:.2f}, New {key}: {new_result[key]:.2f}, Difference: {new_result[key] - value}")

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir
    )

    print(f"Saving TensorBoard log files to: {log_dir}")

    return tensorboard_callback

def get_line(file_name):
    with open(file_name, "r") as f:
        return f.readlines()
    
def preprocess_text_with_line_number(file_name):
    input_line = get_line(file_name)
    abstract_line = ""
    abstract_sample = []
    
    for line in input_line:
        ### New id
        if line.startswith("###"):
            abstract_id = line
            abstract_line = "" # Reset before proceed to next
        ### Start processing the data
        elif line.isspace():
            abstract_line_split = abstract_line.splitlines()

            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                
                target_text_split = abstract_line.split("\t")
                
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_line"] = len(abstract_line_split) - 1

                abstract_sample.append(line_data)
        ### Collecting/ merging the data per id
        else:
            abstract_line += line
  
    return abstract_sample