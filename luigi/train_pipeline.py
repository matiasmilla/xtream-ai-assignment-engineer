import luigi
import pandas as pd
from sklearn.preprocessing import TargetEncoder
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    luigi.run()
