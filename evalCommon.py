import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc


class evalCommon():
    def __init__(self, label, pred_proba):
        super().__init__()
        self.pred_proba = pred_proba
        self.label = label
        self.new_threshold = 0.5


    def evaluation(self):
        # 默认阈值为0.5直接输出结果
        # y_pred = self.model.predict(self.x_test)
        torch.save(self.label, "./Strokeformer3_label1")
        torch.save(self.pred_proba, "./Strokeformer3_pred1")

        # 调整阈值
        
        y_pred = (self.pred_proba >= self.new_threshold).astype(int)
        fpr, tpr, thresholds = roc_curve(self.label, self.pred_proba)
        roc_auc = round(auc(fpr, tpr), 4)
        metrics = {
            'accuracy': round(accuracy_score(self.label, y_pred), 4),
            'recall': round(recall_score(self.label, y_pred), 4),
            'precision': round(precision_score(self.label, y_pred), 4),
            'f1': round(f1_score(self.label, y_pred), 4),
            'roc_auc': roc_auc,
        }
        return metrics