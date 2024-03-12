import luigi
import pandas as pd

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

if __name__ == "__main__":
    luigi.run()
