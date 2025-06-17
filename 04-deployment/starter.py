import argparse
import pickle
import pandas as pd

def read_transform_data(filename):
	df = pd.read_parquet(filename)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df['duration'] = df.duration.dt.total_seconds() / 60

	df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

	categorical = ['PULocationID', 'DOLocationID']
	df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
	
	return df


def apply_model(model, df):
	with open(model, 'rb') as f_in:
		dv, model = pickle.load(f_in)
	
	categorical = ['PULocationID', 'DOLocationID']
	dicts = df[categorical].to_dict(orient='records')
	
	X = dv.transform(dicts)
	y_pred = model.predict(X)
	
	return y_pred

def save_predictions(y_pred, year, month):
	df_result = pd.DataFrame()
	df_result['prediction'] = y_pred
	df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

	output_file = f"yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet"

	df_result.to_parquet(
		output_file,
		engine='pyarrow',
		compression=None,
		index=False
	)
	return output_file

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	
	parser.add_argument('--year', type=int, default=2023, help='Year of the trip data')
	parser.add_argument('--month', type=int, default=3, help='Month of the trip data')
	parser.add_argument('--model', type=str, default='model.bin', help='Path to the model file')
	
	args = parser.parse_args()
	model = args.model
	input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'

	print(f"Loading input file: {input_file}")
	print(f"Loading model file: {model}")

	df = read_transform_data(input_file)

	y_pred = apply_model(model, df)
	print(f"The mean predicted duration is {y_pred.mean():.2f} minutes")

	output_file = save_predictions(y_pred, args.year, args.month)
	print(f"Predictions saved to {output_file}")

