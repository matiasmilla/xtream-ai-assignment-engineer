import luigi
import pandas as pd
from sklearn.preprocessing import TargetEncoder
import joblib

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

if __name__ == "__main__":
    luigi.run()
