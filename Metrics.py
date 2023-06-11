class Metrics:
    def __init__(self, mse, rmse, logloss, mean_per_class_error, r2, accuracy):
        self.mse = mse
        self.rmse = rmse
        self.logloss = logloss
        self.mean_per_class_error = mean_per_class_error
        self.r2 = r2
        self.accuracy = accuracy
    
    def __str__(self):
        return f"<MSE: {self.mse}, RMSE: {self.rmse}, Log Loss: {self.logloss}, Mean Per Class Error: {self.mean_per_class_error}, R^2 Score: {self.r2}, Accuracy: {self.accuracy}>"