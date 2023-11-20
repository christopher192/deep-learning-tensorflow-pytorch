import tensorflow as tf
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

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

def plot_loss_curve(history): 
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss curve
    plt.plot(epochs, loss, label = 'training_loss')
    plt.plot(epochs, val_loss, label = 'val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy curve
    plt.figure()
    plt.plot(epochs, accuracy, label = 'training_accuracy')
    plt.plot(epochs, val_accuracy, label = 'val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

def unzip_data(file_name):
    zip_ref = zipfile.ZipFile(file_name, "r")
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def compare_history(original_history, new_history, initial_epochs = 5):
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize = (8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label = 'Training Accuracy')
    plt.plot(total_val_acc, label = 'Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
        plt.ylim(), label = 'Start Fine Tuning')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label = 'Training Loss')
    plt.plot(total_val_loss, label = 'Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
        plt.ylim(), label = 'Start Fine Tuning')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.to("cpu")
    x, y = x.to("cpu"), y.to("cpu")

    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    x_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    
    with torch.inference_mode():
        y_logit = model(x_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logit, dim = 1).argmax(dim = 1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logit))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(x[:, 0], x[:, 1], c = y, s = 40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
def plot_prediction(train_data, train_label, test_data, test_label, prediction = None):
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_label, c = "b", s = 4, label = "Training data")
    plt.scatter(test_data, test_label, c = "g", s = 4, label = "Testing data")

    if prediction is not None:
        plt.scatter(test_data, prediction, c = "r", s = 4, label = "Prediction")

    plt.legend(prop = {"size" : 14})
    
def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    
    return acc

def pytorch_plot_loss_curve(result): 
    train_loss = result["train_loss"]
    test_loss = result["test_loss"]

    train_accuracy = result["train_accuracy"]
    test_accuracy = result["test_accuracy"]

    epoch_number = range(len(result["train_loss"]))

    plt.figure(figsize = (15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_number, train_loss, label = "train_loss")
    plt.plot(epoch_number, test_loss, label = "test_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_number, train_accuracy, label = "train_accuracy")
    plt.plot(epoch_number, test_accuracy, label = "test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()