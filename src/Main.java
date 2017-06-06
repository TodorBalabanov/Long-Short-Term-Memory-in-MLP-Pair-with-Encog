import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

/**
 * Single entry point.
 * 
 * @author Todor Balabanov
 */
public class Main {
	private static final String DATA_FILE_NAME = "./dat/number-of-earthquakes-per-year-m.csv";

	/**
	 * Statistical significance is dependent of the experiments number.
	 */
	private static final int NUMBER_OF_EXPERIMENTS = 30;

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
		try {
			List<Double> values = new ArrayList<Double>();

			/*
			 * Read raw data.
			 */
			ReadCSV csv = new ReadCSV(new FileInputStream(DATA_FILE_NAME), true, CSVFormat.DECIMAL_POINT);
			while (csv.next() == true) {
				values.add(csv.getDouble(1));
			}
			csv.close();

			double min = values.stream().min(Double::compare).get();
			double max = values.stream().max(Double::compare).get();

			/*
			 * Split to training examples.
			 */
			for (int i = 0; i < values.size() - LEAD_SIZE; i++) {
				for (int j = 0; j < LAG_SIZE; j++) {
					// TODO Add normalized input data.
				}
				for (int j = 0; j < LEAD_SIZE; j++) {
					// TODO Add normalized output data.
				}
			}
		} catch (FileNotFoundException exception) {
			System.err.println(exception);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("Start ...");

		System.out.println("Stop ...");
	}
}
