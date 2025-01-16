from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(all_predictions, all_labels, mode):
    # Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="binary")
    recall = recall_score(all_labels, all_predictions, average="binary")
    f1 = f1_score(all_labels, all_predictions, average="binary")
    accuracy, precision, recall, f1 = round(accuracy,4), round(precision,4), round(recall,4), round(f1,4) 
    print(f"{mode}: - Accuracy: {accuracy:.4f} Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return {"accuracy": accuracy,"precision": precision, "recall": recall, "f1": f1}