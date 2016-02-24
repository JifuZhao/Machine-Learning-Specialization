package cs446.homework2;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;

public class fullDecisionTree {

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err.println("Usage: WekaTester arff-file");
	    System.exit(-1);
	}

	// Load the data
	Instances train = new Instances(new FileReader(new File(args[0]))); // modified
	Instances test = new Instances(new FileReader(new File(args[1]))); // modified

	// The last attribute is the class label
	train.setClassIndex(train.numAttributes() - 1); //modified
	test.setClassIndex(test.numAttributes() - 1); //modified

	// Create a new ID3 classifier. This is the modified one where you can
	// set the depth of the tree.
	Id3 classifier = new Id3();

	// An example depth. If this value is -1, then the tree is grown to full
	// depth.
	classifier.setMaxDepth(-1);

	// Train
	classifier.buildClassifier(train);

	// Print the classfier
	System.out.println(classifier);
	System.out.println();

	// Evaluate on the test set
	Evaluation evaluation = new Evaluation(test);
	evaluation.evaluateModel(classifier, test);
	System.out.println(evaluation.toSummaryString());

    }
}
