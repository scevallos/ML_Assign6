package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

// TODO: FIX ME SO THAT I'M NOT JUST THE PERCEPTRON!

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Maria Martinez & Sebastian Cevallos
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	// Parameters to keep track of
	private int loss;
	private int reg;
	private double lambda;
	private double eta;

	// TODO: Zero parameter constructor: Just don't have anything, or explicitly
	// include empty constructor? Could set defaults above /w no constructor
	public GradientDescentClassifier() {
		// Default behavior below
		loss = this.EXPONENTIAL_LOSS;
		reg = this.NO_REGULARIZATION;
		lambda = 0.01;
		eta = 0.01;
	}

	/**
	 * Takes an int and selects the loss function to use (based on the
	 * constants)
	 * 
	 * @param lossType
	 * @throws InvalidParameterException
	 */
	public void setLoss(int lossType) throws InvalidParameterException {
		// Check that it's valid first (between 0 & 1 inclusive)
		if (lossType > -1 && lossType < 2)
			loss = lossType;
		else
			throw new InvalidParameterException("Invalid loss type chosen!");
	}

	/**
	 * Takes an int and selects the regularization method to use
	 * 
	 * @param regType
	 * @throws InvalidParameterException
	 */
	public void setRegularization(int regType) throws InvalidParameterException {
		// Check that it's valid first (between 0 & 2 inclusive)
		if (regType > -1 && regType < 3)
			reg = regType;
		else
			throw new InvalidParameterException(
					"Invalid regularization type chosen!");

	}

	/**
	 * Takes a double and sets that as the new lambda to use.
	 * 
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		// TODO is there a valid/invalid range? Like lambda can't be negative?
		this.lambda = lambda;
	}

	/**
	 * Takes a double and sets that as the new eta to use.
	 * 
	 * @param eta
	 */
	public void setEta(double eta) {
		// TODO is there a valid/invalid range?
		this.eta = eta;
	}

	/**
	 * TODO: Comment
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData()
				.clone();

		for (int it = 0; it < 1; it++) {
			// Collections.shuffle(training);

			for (Example e : training) {
				// Removed check to see if wrong bc GD always updates
				double label = e.getLabel();

				double dotProduct = this.getDistanceFromHyperplane(e, weights,
						b);

				double lossVal = 0.0;
				if (loss == this.EXPONENTIAL_LOSS) {
					lossVal = Math.exp((-label) * (dotProduct));
				} else
					// loss == this.HINGE_LOSS
					lossVal = Math.max(0.0, (1 - (label * dotProduct)));

				// update the weights
				double r = 0.0;
				for (Integer featureIndex : e.getFeatureSet()) {
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);

					r = 0.0;
					// if no regularization, just leave as zero
					if (reg == this.L1_REGULARIZATION)
						r = lambda * Math.signum(oldWeight);

					else if (reg == this.L2_REGULARIZATION)
						r = lambda * oldWeight;

					// System.out.println(oldWeight + this.eta * label *
					// featureValue * lossVal - r);
					weights.put(featureIndex, oldWeight + eta
							* (label * featureValue * lossVal - lambda * r));
				}

				double r2 = 0.0; // if no regularization, just leave as zero
				if (reg == this.L1_REGULARIZATION)
					r2 = lambda * Math.signum(b);
				else if (reg == this.L2_REGULARIZATION)
					r2 = lambda * b;

				// update b
				b += eta * (label * lossVal - lambda * r2);

			}
		}
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e
	 *            the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and
	 * inputB
	 * 
	 * @param e
	 *            example to predict
	 * @param w
	 *            the set of weights to use
	 * @param inputB
	 *            the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}

	public static void main(String[] args) throws InvalidParameterException {
		// Collect the data
		String titanic = "/home/scevallos/Documents/ML/titanic-train.perc.csv";
//		String titanic = "/home/scevallos/Documents/ML/data/SPECT.train";
		// String tr = "/home/scevallos/Documents/ML/Assign6/train.csv";
		// String te = "/home/scevallos/Documents/ML/Assign6/test.csv";

		// DataSet train = new DataSet(tr, DataSet.CSVFILE);
		// DataSet test = new DataSet(te, DataSet.CSVFILE);
		DataSet data = new DataSet(titanic, DataSet.CSVFILE);
		System.out.println("Loaded in DataSet!");

		CrossValidationSet cvSet = new CrossValidationSet(data, 10);

		DataSetSplit split = data.split(0.8);

		double lambda = 0.1;
		double eta = 0.1;

		// Prepare the Classifier
		GradientDescentClassifier GDC = new GradientDescentClassifier();
		GDC.setRegularization(L2_REGULARIZATION);
		GDC.setLoss(HINGE_LOSS);

		ArrayList<Double> avgs = new ArrayList<Double>();
		ArrayList<Double> etas = new ArrayList<Double>();
		ArrayList<Double> lams = new ArrayList<Double>();
		
		ArrayList<Double> pcents = new ArrayList<Double>();
		
		for (int i = 0; i < 10; i++) {
			
			lams.add(lambda);
			etas.add(eta);
			
			GDC.setLambda(lambda);
			GDC.setEta(eta);
			
			GDC.train(split.getTrain());
			
			double correct = 0.0;
			for (Example e: split.getTest().getData()){
				double pred = GDC.classify(e);
				double label = e.getLabel();
				if (label == pred)
					correct++;
			}
			
			double percent = correct/split.getTest().getData().size();
			pcents.add(percent);
//			double correct = 0.0;
//			double avg = 0.0;
//			for (int splitNum = 0; splitNum < 10; splitNum++) {
//				correct = 0.0;
//				DataSetSplit temp = cvSet.getValidationSet(splitNum);
//				// set hyper-parameters
//				GDC.train(temp.getTrain());
//
//				for (Example e : temp.getTest().getData()) {
//					double pred = GDC.classify(e);
//					if (pred == e.getLabel())
//						correct++;
//				}
//				pcents.add("" + correct / (temp.getTest().getData().size()));
//				avg += correct / (temp.getTest().getData().size());
//			}
//			avgs.add(avg/10);
//			avg = 0.0;
			
//			lambda *= 0.9;
			eta *= 0.9;
			
		}
		int maxIndex = 0;
		for(int i = 1; i < pcents.size(); i++){
			if (pcents.get(i) > pcents.get(maxIndex))
				maxIndex = i;
		}
		System.out.println("max val: " + pcents.get(maxIndex));
		System.out.println("max val index: " + maxIndex);
		System.out.println("param value: " + etas.get(maxIndex));
		
		System.out.println(pcents);
		
		// System.out.println(GDC.weights);

		// System.out.println("# Correct: " + correct);
		// System.out.println("# Total: " + test.getData().size());
		//
		// System.out.println("Percent Correct: " + correct
		// / test.getData().size());

	}
}
