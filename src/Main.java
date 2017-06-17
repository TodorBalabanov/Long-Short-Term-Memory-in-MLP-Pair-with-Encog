import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizeArray;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

/**
 * Single entry point.
 * 
 * @author Todor Balabanov
 */
public class Main {
	private static final String DATA_FILE_NAME = "../dat/number-of-earthquakes-per-year-m.csv";

	/**
	 * Statistical significance is dependent of the experiments number.
	 */
	private static final int NUMBER_OF_EXPERIMENTS = 30;

	/**
	 * Time limit for training.
	 */
	private static final long MAX_TRAINING_TIME = 1 * 60 * 1000;

	/**
	 * Data collection time interval.
	 */
	private static final long SINGLE_MEASUREMENT_MILLISECONDS = 6 * 1000;

	/**
	 * Lag frame of the time series data window.
	 */
	private static final int LAG_SIZE = 15;

	/**
	 * Lead frame of the time series data window.
	 */
	private static final int LEAD_SIZE = 5;

	/**
	 * Single MLP hidden layer size. Size of the hidden layer is subject of
	 * experiments.
	 */
	private static final int MLP_HIDDEN_SIZE = (LAG_SIZE + LEAD_SIZE + 1) / 2;

	/**
	 * MLP pair hidden layer size. Size of the hidden layer is subject of
	 * experiments.
	 */
	private static final int PAIR_HIDDEN_SIZE = (LEAD_SIZE + LEAD_SIZE + 1) / 2;

	/**
	 * Data sat in the range 0.0 to +1.0.
	 */
	private static final NeuralDataSet ZERO_ONE_DATA = new BasicNeuralDataSet();

	/**
	 * Data sat in the range -1.0 to +1.0.
	 */
	private static final NeuralDataSet MINUS_PLUS_ONE_DATA = new BasicNeuralDataSet();

	/**
	 * Single MLP network.
	 */
	private static BasicNetwork single = new BasicNetwork();

	/**
	 * MLP pair network.
	 */
	private static BasicNetwork pair[] = { new BasicNetwork(), new BasicNetwork() };

	static {
		/*
		 * Single MLP network construction.
		 */
		single.addLayer(new BasicLayer(new ActivationSigmoid(), true, LAG_SIZE));
		single.addLayer(new BasicLayer(new ActivationSigmoid(), true, MLP_HIDDEN_SIZE));
		single.addLayer(new BasicLayer(new ActivationSigmoid(), false, LEAD_SIZE));
		single.getStructure().finalizeStructure();
		single.reset();

		/*
		 * MLP network construction. The basic MLP has bigger input, because it
		 * takes the output of the secondary MLP. The secondary MLP has equal
		 * input and out put size. It is working as a repeater.
		 */
		pair[0].addLayer(new BasicLayer(new ActivationSigmoid(), true, LAG_SIZE + LEAD_SIZE));
		pair[0].addLayer(new BasicLayer(new ActivationSigmoid(), true, MLP_HIDDEN_SIZE));
		pair[0].addLayer(new BasicLayer(new ActivationSigmoid(), false, LEAD_SIZE));
		pair[0].getStructure().finalizeStructure();
		pair[0].reset();
		pair[1].addLayer(new BasicLayer(new ActivationSigmoid(), true, LEAD_SIZE));
		pair[1].addLayer(new BasicLayer(new ActivationSigmoid(), true, PAIR_HIDDEN_SIZE));
		pair[1].addLayer(new BasicLayer(new ActivationSigmoid(), false, LEAD_SIZE));
		pair[1].getStructure().finalizeStructure();
		pair[1].reset();

		/*
		 * It is used for time measurement calibration.
		 */
		try {
			(new ResilientPropagation(single, new BasicNeuralDataSet())).iteration();
			(new ResilientPropagation(pair[0], new BasicNeuralDataSet())).iteration();
			(new ResilientPropagation(pair[1], new BasicNeuralDataSet())).iteration();
		} catch (Exception e) {
		}

		/*
		 * Read experimental data and form data sets.
		 */
		List<Double> values = new ArrayList<Double>();
		try {
			/*
			 * Read raw data.
			 */
			ReadCSV csv = new ReadCSV(new FileInputStream(DATA_FILE_NAME), true, CSVFormat.DECIMAL_POINT);
			while (csv.next() == true) {
				values.add(csv.getDouble(1));
			}
			csv.close();
		} catch (FileNotFoundException exception) {
			System.err.println(exception);
		}

		NormalizeArray normalizer = new NormalizeArray();

		/*
		 * Normalization for the sigmoid function.
		 */
		normalizer.setNormalizedLow(0.1);
		normalizer.setNormalizedHigh(0.9);
		double zeroOneNormalized[] = normalizer.process(values.stream().mapToDouble(Double::doubleValue).toArray());

		/*
		 * Normalization for the hyperbolic tangent function.
		 */
		normalizer.setNormalizedLow(-0.9);
		normalizer.setNormalizedHigh(+0.9);
		double minusOneOneNormalized[] = normalizer.process(values.stream().mapToDouble(Double::doubleValue).toArray());

		/*
		 * Split to zero-one training examples.
		 */
		for (int i = 0; i < values.size() - (LAG_SIZE + LEAD_SIZE); i++) {
			MLData input = new BasicMLData(LAG_SIZE);
			MLData ideal = new BasicMLData(LEAD_SIZE);
			MLDataPair pair = new BasicMLDataPair(input, ideal);

			for (int j = 0; j < LAG_SIZE; j++) {
				input.setData(j, zeroOneNormalized[i + j]);
			}
			for (int j = 0; j < LEAD_SIZE; j++) {
				ideal.setData(j, zeroOneNormalized[i + LAG_SIZE + j]);
			}

			ZERO_ONE_DATA.add(pair);
		}

		/*
		 * Split to minus-plus-one training examples.
		 */
		for (int i = 0; i < values.size() - (LAG_SIZE + LEAD_SIZE); i++) {
			MLData input = new BasicMLData(LAG_SIZE);
			MLData ideal = new BasicMLData(LEAD_SIZE);
			MLDataPair pair = new BasicMLDataPair(input, ideal);

			for (int j = 0; j < LAG_SIZE; j++) {
				input.setData(j, minusOneOneNormalized[i + j]);
			}
			for (int j = 0; j < LEAD_SIZE; j++) {
				ideal.setData(j, minusOneOneNormalized[i + LAG_SIZE + j]);
			}

			MINUS_PLUS_ONE_DATA.add(pair);
		}
	}

	/**
	 * First training experiment.
	 * 
	 * @return Statistics collected during the training process.
	 */
	private static List<Object> train1() {
		List<Object> result = new ArrayList<>();

		single.reset();
		final Train train = new ResilientPropagation(single, ZERO_ONE_DATA);

		/*
		 * Initial record.
		 */ {
			train.iteration();
			Object record[] = { Double.valueOf(train.getError()), Long.valueOf(SINGLE_MEASUREMENT_MILLISECONDS),
					Long.valueOf(0) };
			/*
			 * result.add(record);
			 */
		}

		int epoch = 0;
		for (long stop = System.currentTimeMillis() + MAX_TRAINING_TIME; System.currentTimeMillis() < stop;) {
			long start = System.currentTimeMillis();

			do {
				train.iteration();

				epoch++;
			} while ((System.currentTimeMillis() - start) < SINGLE_MEASUREMENT_MILLISECONDS);

			Object record[] = { Double.valueOf(train.getError()), Long.valueOf((System.currentTimeMillis() - start)),
					Long.valueOf(epoch) };

			result.add(record);
		}

		return result;
	}

	/**
	 * Second training experiment.
	 * 
	 * @return Statistics collected during the training process.
	 */
	private static List<Object> train2() {
		List<Object> result = new ArrayList<>();

		pair[0].reset();
		pair[1].reset();

		Train train[] = { null, null };

		/*
		 * Initial record.
		 */ {
			/*
			 * Form training data.
			 */
			MLDataSet[] data = { new BasicNeuralDataSet(), new BasicNeuralDataSet() };
			for (int i = 0; i < ZERO_ONE_DATA.size(); i++) {
				double output[] = new double[LEAD_SIZE];

				double feedback[] = new double[LEAD_SIZE];
				if (i - 1 >= 0) {
					feedback = ZERO_ONE_DATA.get(i - 1).getIdeal().getData();
				} else {
					// TODO May be it is better to initialize with other value.
					feedback = new Random().doubles(LEAD_SIZE, 0.0, 1.0).toArray();
				}

				/*
				 * Request secondary MLP for output in order to supply it in the
				 * input of the primary MLP.
				 */
				pair[1].compute(feedback, output);

				/*
				 * Concatenate time series input data with secondary MLP output.
				 */
				double[] input = new double[ZERO_ONE_DATA.get(i).getInput().getData().length + output.length];
				System.arraycopy(ZERO_ONE_DATA.get(i).getInput().getData(), 0, input, 0,
						ZERO_ONE_DATA.get(i).getInput().getData().length);
				System.arraycopy(output, 0, input, ZERO_ONE_DATA.get(i).getInput().getData().length, output.length);

				/*
				 * Add training example for the primary MLP.
				 */
				data[0].add(new BasicMLDataPair(new BasicMLData(input), ZERO_ONE_DATA.get(i).getIdeal()));

				/*
				 * Generate output of the primary MLP in order to supply this
				 * values as input for the secondary MLP.
				 */
				pair[0].compute(input, output);

				/*
				 * Add training example for the secondary MLP.
				 */
				data[1].add(new BasicMLDataPair(new BasicMLData(output), ZERO_ONE_DATA.get(i).getIdeal()));
			}

			train = new Train[] { new ResilientPropagation(pair[0], data[0]),
					new ResilientPropagation(pair[1], data[1]) };

			train[0].iteration();
			train[1].iteration();

			Object record[] = { Double.valueOf(train[0].getError()), Long.valueOf(SINGLE_MEASUREMENT_MILLISECONDS),
					Long.valueOf(0) };

			/*
			 * result.add(record);
			 */
		}

		/*
		 * Training.
		 */
		int epoch = 0;
		for (long stop = System.currentTimeMillis() + MAX_TRAINING_TIME; System.currentTimeMillis() < stop;) {
			long start = System.currentTimeMillis();

			do {
				/*
				 * Form training data.
				 */
				MLDataSet[] data = { new BasicNeuralDataSet(), new BasicNeuralDataSet() };
				for (int i = 0; i < ZERO_ONE_DATA.size(); i++) {
					double output[] = new double[LEAD_SIZE];

					double feedback[] = new double[LEAD_SIZE];
					if (i - 1 >= 0) {
						feedback = ZERO_ONE_DATA.get(i - 1).getIdeal().getData();
					} else {
						// TODO May be it is better to initialize with other
						// value.
						feedback = new Random().doubles(LEAD_SIZE, 0.0, 1.0).toArray();
					}

					/*
					 * Request secondary MLP for output in order to supply it in
					 * the input of the primary MLP.
					 */
					pair[1].compute(feedback, output);

					/*
					 * Concatenate time series input data with secondary MLP
					 * output.
					 */
					double[] input = new double[ZERO_ONE_DATA.get(i).getInput().getData().length + output.length];
					System.arraycopy(ZERO_ONE_DATA.get(i).getInput().getData(), 0, input, 0,
							ZERO_ONE_DATA.get(i).getInput().getData().length);
					System.arraycopy(output, 0, input, ZERO_ONE_DATA.get(i).getInput().getData().length, output.length);

					/*
					 * Add training example for the primary MLP.
					 */
					data[0].add(new BasicMLDataPair(new BasicMLData(input), ZERO_ONE_DATA.get(i).getIdeal()));

					/*
					 * Generate output of the primary MLP in order to supply
					 * this values as input for the secondary MLP.
					 */
					pair[0].compute(input, output);

					/*
					 * Add training example for the secondary MLP.
					 */
					data[1].add(new BasicMLDataPair(new BasicMLData(output), ZERO_ONE_DATA.get(i).getIdeal()));
				}

				train = new Train[] { new ResilientPropagation(pair[0], data[0]),
						new ResilientPropagation(pair[1], data[1]) };

				train[0].iteration();
				train[1].iteration();

				epoch++;
			} while ((System.currentTimeMillis() - start) < SINGLE_MEASUREMENT_MILLISECONDS);

			Object record[] = { Double.valueOf(train[0].getError()), Long.valueOf((System.currentTimeMillis() - start)),
					Long.valueOf(epoch) };

			result.add(record);
		}

		return result;
	}

	/**
	 * Print results.
	 * 
	 * @param statistics
	 *            Results.
	 */
	private static void print(List<List<Object>> statistics) {
		String texts[] = new String[statistics.get(0).size()];

		for (int i = 0; i < texts.length; i++) {
			texts[i] = "";
		}

		loop: for (int c = 0;; c++) {
			for (List<Object> experiment : statistics) {
				int i = 0;
				for (Object record : experiment) {
					if (c >= ((Object[]) record).length) {
						break loop;
					}

					texts[i] = texts[i] + "\t" + ((Object[]) record)[c];
					i++;
				}
			}
		}

		for (int i = 0; i < texts.length; i++) {
			System.out.println(texts[i].trim());
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.err.println("Start ...");

		List<List<Object>> statistics = new ArrayList<List<Object>>();

		statistics.clear();
		System.err.println("First ...");
		for (int g = 0; g < NUMBER_OF_EXPERIMENTS; g++) {
			statistics.add(train1());
			System.err.print("*");
		}
		print(statistics);

		statistics.clear();
		System.err.println("Second ...");
		for (int g = 0; g < NUMBER_OF_EXPERIMENTS; g++) {
			statistics.add(train2());
			System.err.print("*");
		}
		print(statistics);

		System.err.println("Stop ...");
	}
}
