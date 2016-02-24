package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.HashSet;

import weka.classifiers.RandomizableClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import cs446.weka.classifiers.trees.Id3;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.core.converters.ArffSaver;

public class SGDgenerator {

// modified based on the featureGenerator.java

    static String[] features;
    private static FastVector zeroOne;
    private static FastVector labels;
    //int N = 100;

    static {
	features = new String[100]; //modified

	List<String> ff = new ArrayList<String>();

	for (int i = 0; i < 100; i++) {
	    ff.add("stumpResult" + "=" + String.valueOf(i));
	}

	features = ff.toArray(new String[ff.size()]);

	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");
    }
    
//    public static Instances readData(String fileName) throws Exception {
//
//	Instances instances = initializeAttributes();
//	Scanner scanner = new Scanner(new File(fileName));
//
//	while (scanner.hasNextLine()) {
//	    String line = scanner.nextLine();
//
//	    Instance instance = makeInstance(instances, line);
//
//	    instances.add(instance);
//	}
//
//	scanner.close();/
//
//	return instances;
 //   }

    private static Instances initializeAttributes() {

	String nameOfDataset = "Badges";

	Instances instances;

	FastVector attributes = new FastVector(9);
	for (String featureName : features) {
	    attributes.addElement(new Attribute(featureName, zeroOne));
	}
	Attribute classLabel = new Attribute("Class", labels);
	attributes.addElement(classLabel);

	instances = new Instances(nameOfDataset, attributes, 0);

	instances.setClass(classLabel);

	return instances;

    }

   
    public static void main(String[] args) throws Exception {

	if (args.length != 4) {
	    System.err.println("Usage: WekaTester arff-file");
	    System.exit(-1);
	}

	// Load the data
	Instances train = new Instances(new FileReader(new File(args[0]))); // modified
	Instances test = new Instances(new FileReader(new File(args[1]))); // modified

	// The last attribute is the class label
	train.setClassIndex(train.numAttributes() - 1); //modified
	test.setClassIndex(test.numAttributes() - 1); //modified
    
    Id3[] Stump = new Id3[100];
    for (int i = 0; i < 100; i++) {
    	// Train on 50% of the data
    	Random rand = new Random();
	    Instances trainData = train.trainCV(2,0, rand); // modified
	    Stump[i] = new Id3();
	    Stump[i].setMaxDepth(4);
	    Stump[i].buildClassifier(trainData);
	    
	}    
	
	Instances trainConvert = initializeAttributes();
	trainConvert.setClassIndex(trainConvert.numAttributes() - 1);
	
	for (int i = 0; i < train.numInstances(); i++) {
	    Instance instanceConvert = new Instance(101);
	    instanceConvert.setDataset(trainConvert);
	    
	    for (int j = 0; j < 100; j++) {
	        if (Stump[j].classifyInstance(train.instance(i)) == 1.0) {
	            instanceConvert.setValue(j, 1.0);
	        }
	        else{
	            instanceConvert.setValue(j, 0.0);
	        }
	        instanceConvert.setClassValue(train.instance(i).classValue());
	    }
	    
	    trainConvert.add(instanceConvert);        
	}   

    Instances testConvert = initializeAttributes();
	testConvert.setClassIndex(testConvert.numAttributes() - 1);
	
	for (int i = 0; i < test.numInstances(); i++) {
	    Instance instanceConvert = new Instance(101);
	    instanceConvert.setDataset(testConvert);
	    
	    for (int j = 0; j < 100; j++) {
	        if (Stump[j].classifyInstance(test.instance(i)) == 1.0) {
	            instanceConvert.setValue(j, 1.0);
	        }
	        else{
	            instanceConvert.setValue(j, 0.0);
	        }
	        instanceConvert.setClassValue(test.instance(i).classValue());
	    } 
	    testConvert.add(instanceConvert);        
	} 

	ArffSaver saver1 = new ArffSaver();
	saver1.setInstances(trainConvert);
	saver1.setFile(new File(args[2]));
	saver1.writeBatch();
	
	ArffSaver saver2 = new ArffSaver();
	saver2.setInstances(testConvert);
	saver2.setFile(new File(args[3]));
	saver2.writeBatch();
	

    }
}
