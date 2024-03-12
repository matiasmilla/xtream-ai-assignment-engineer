import luigi
import pandas as pd
from sklearn.preprocessing import TargetEncoder, StandardScaler
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class CleanData(luigi.Task):
    input_file = luigi.Parameter()
    output_file = luigi.Parameter(default='dataCleaned.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.df['price'] > 0]
        self.df = self.df[(self.df['x'] > 0) & (self.df['y'] > 0) & (self.df['z'] > 0)]
        self.df.to_csv(self.output_file, index=False)

class FeatureCreation(luigi.Task):
    input_file = luigi.Parameter()
    output_file = luigi.Parameter(default='dataWithSize.csv')

    def requires(self):
        return CleanData(self.input_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.df = pd.read_csv(self.input().path)
        self.df['size'] = self.df['x']*self.df['y']*self.df['z']
        self.df.to_csv(self.output_file, index=False)

class CategoricalEncoding(luigi.Task):
    input_file = luigi.Parameter()
    encoder_file = luigi.Parameter(default='encoder.sav')
    output_file = luigi.Parameter(default='encodedData.csv')

    def requires(self):
        return FeatureCreation(self.input_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.df = pd.read_csv(self.input().path)
        categorical_columns = ['color', 'clarity']
        target_encoder = TargetEncoder(target_type='continuous', smooth="auto", cv=10)
        self.df[categorical_columns] = target_encoder.fit_transform(self.df[categorical_columns], self.df['price'])
        self.df.to_csv(self.output_file, index=False)
        joblib.dump(target_encoder, self.encoder_file)

class FeatureSelection(luigi.Task):
    input_file = luigi.Parameter()
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CategoricalEncoding(self.input_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.df = pd.read_csv(self.input().path)
        features = ['size', 'color', 'clarity']
        self.df = self.df[features]
        self.df.to_csv(self.output_file, index=False)

class GetTarget(luigi.Task):
    input_file = luigi.Parameter()
    target_name = luigi.Parameter(default='price')
    output_file = luigi.Parameter(default='target.csv')

    def requires(self):
        return CategoricalEncoding(self.input_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        self.df = pd.read_csv(self.input().path)
        self.target = np.log(self.df[self.target_name])
        self.target.to_csv(self.output_file, sep=',', index=False)

class DataSplitting(luigi.Task):
    input_file = luigi.Parameter()
    X_train_file = luigi.Parameter(default='X_train.csv')
    X_test_file = luigi.Parameter(default='X_test.csv')
    y_train_file = luigi.Parameter(default='y_train.csv')
    y_test_file = luigi.Parameter(default='y_test.csv')

    def requires(self):
        return {"features": FeatureSelection(self.input_file), "target": GetTarget(self.input_file)}

    def output(self):
        return {'X_train_file': luigi.LocalTarget(self.X_train_file), 'X_test_file': luigi.LocalTarget(self.X_test_file),
                'y_train_file': luigi.LocalTarget(self.y_train_file), 'y_test_file': luigi.LocalTarget(self.y_test_file)}

    def run(self):
        X = pd.read_csv(self.input()["features"].path)
        y = pd.read_csv(self.input()["target"].path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84)
        X_train.to_csv(self.X_train_file, index=False)
        X_test.to_csv(self.X_test_file, index=False)
        y_train.to_csv(self.y_train_file, index=False)
        y_test.to_csv(self.y_test_file, index=False)

class FeatureScaling(luigi.Task):
    input_file = luigi.Parameter()
    X_train_file = luigi.Parameter(default='X_train.csv')
    X_test_file = luigi.Parameter(default='X_test.csv')
    scaler_file = luigi.Parameter(default='scaler.sav')

    def requires(self):
        return DataSplitting(self.input_file)

    def output(self):
        return {'X_train_file': luigi.LocalTarget(self.X_train_file), 'X_test_file': luigi.LocalTarget(self.X_test_file)}

    def run(self):
        scaler = StandardScaler()
        X_train = pd.read_csv(self.input()['X_train_file'].path)
        X_test = pd.read_csv(self.input()['X_test_file'].path)
        X_train = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns = ['size', 'color', 'clarity'])
        X_test = scaler.fit_transform(X_test)
        X_test = pd.DataFrame(X_test, columns = ['size', 'color', 'clarity'])
        X_train.to_csv(self.X_train_file, index=False)
        X_test.to_csv(self.X_test_file, index=False)
        joblib.dump(scaler, self.scaler_file)

class TrainModel(luigi.Task):
    input_file = luigi.Parameter()
    model_file = luigi.Parameter(default='diamonds_model.json')

    def requires(self):
        return {"features": FeatureScaling(self.input_file), "target": DataSplitting(self.input_file)}

    def output(self):
        return luigi.LocalTarget(self.model_file)

    def run(self):
        X_train = pd.read_csv(self.input()["features"]["X_train_file"].path)
        y_train = pd.read_csv(self.input()["target"]["y_train_file"].path)
        self.model = XGBRegressor(min_child_weight=1, max_depth=8, learning_rate=0.1, gamma=0.0)
        self.model.fit(X_train, y_train)
        self.model.save_model(self.model_file)

if __name__ == "__main__":
    luigi.run()
