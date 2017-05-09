package au.edu.usyd.it.yangpy.snp;

/* 
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
*/

import java.util.LinkedList;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.core.neighboursearch.*;
import weka.classifiers.trees.*;
import weka.classifiers.evaluation.auc.AUCCalculator;
import weka.classifiers.evaluation.auc.ClassSort;
import weka.classifiers.lazy.*;


/** ----------------
* Creation Date: 
* 01/05/2007
* 
* Modification Date:
* 03/09/2010
* 09/11/2010
*/

public class Ensemble {

	private Instances train;
	private Instances test;
	private double[] givenValue;		// prediction results variables.
	private int numInstances;
	private int numClasses;
	private int numClassifiers = 5;
	private double[] aucClassifiers = new double[numClassifiers];
	// each classifier will give a predict distribution for each instance
	private double[][][] predictDistribution;
	// each classifier will give a predict value for each instance
	private double[][] predictValue;
	// vote value is given by combining the predict distribution
	private double[][] voteValue;
	
	public Ensemble (Instances train, Instances test) {
		this.train = train;
		this.test = test;
	}
	
	// this method evaluate classifiers 
	public void ensemble(String mode) throws Exception{
		
		numInstances = test.numInstances();
		numClasses = test.numClasses();
		givenValue = new double[numInstances];
		predictDistribution = new double[numClassifiers][numInstances][numClasses];
		predictValue = new double[numClassifiers][numInstances];
		voteValue = new double[numInstances][numClasses];
		
		// Setting the given class values of the test instances.
		for (int i = 0; i < numInstances; i++) {
			givenValue[i] = test.instance(i).classValue();
		}
		
		// Calculating the predicted class values using each classifier respectively.
		// J48 coverTree1NN KStar coverTree3NN coverTree5NN
		
		J48 tree = new J48();
		tree.setUnpruned(true);
		aucClassifiers[0] = classify(tree, 0);
		
		KStar kstar = new KStar();
		aucClassifiers[1] = classify(kstar, 1);

		IBk ctnn1 = new IBk(1);	
		CoverTree search = new CoverTree();
		ctnn1.setNearestNeighbourSearchAlgorithm(search);
		aucClassifiers[2] = classify(ctnn1, 2);

		IBk ctnn3 = new IBk(3);	
		ctnn3.setNearestNeighbourSearchAlgorithm(search);
		aucClassifiers[3] = classify(ctnn3, 3);

		IBk ctnn5 = new IBk(5);	
		ctnn5.setNearestNeighbourSearchAlgorithm(search);
		aucClassifiers[4] = classify(ctnn5, 4);		
		
		// Print the classification results if in print mode.
		if(mode.equals("v")){
			System.out.println("J48   AUC: " + aucClassifiers[0]);
			System.out.println("KStar AUC: " + aucClassifiers[1]);
			System.out.println("CTNN1 AUC: " + aucClassifiers[2]);
			System.out.println("CTNN3 AUC: " + aucClassifiers[3]);
			System.out.println("CTNN5 AUC: " + aucClassifiers[4]);
			System.out.println("	-			-	");
		}
	}

	// blocking integration
	public double blocking() {
		double blockAccuracy = 0.0;
		
		// Calculating blocking accuracy (average accuracy).
		for (int i = 0; i < numClassifiers; i++) {
			blockAccuracy += aucClassifiers[i];
		}
		
		blockAccuracy /= numClassifiers; 
		
		return blockAccuracy;
	}

	// voting integration 
	public double voting () {

		for (int i = 0; i < test.numInstances(); i++) {
			// combine prediction
			for (int cId = 0; cId < numClassifiers; cId++) {
				for (int t = 0; t < test.numClasses(); t++) {
					voteValue[i][t] += predictDistribution[cId][i][t];
				}
			}
			
			for (int t = 0; t < test.numClasses(); t++) {
				voteValue[i][t] /= numClassifiers;
			}
		}
		
		int classIndex = 1;
		double[][] probList = new double[givenValue.length][2];
		for (int i = 0; i < givenValue.length; i++){
			probList[i][0] = voteValue[i][classIndex];
			probList[i][1] = givenValue[i];
		}
		
		double aucEnsemble = areaUnderROC(probList, classIndex);
		return aucEnsemble * 100;
	}	

	// kappa diversity
	public double kappaDiversity () {
		double diversityScore = 0.0;
		double phi_1 = 0.0;
		double phi_2 = 0.0;
		int count = 0;
		
		for(int i = 0; i < numClassifiers; i++){
			for(int j=i+1; j<numClassifiers; j++){
				count++;
				double a0=0;
				double a1=0;
				double b0=0;
				double b1=0;
				int numOfAgree = 0;
				phi_1 = 0.0;
				phi_2 = 0.0;
				
				for(int k = 0; k < numInstances; k++){
					//phi_1 = classifier[i] and classifier[j] agree / N
					if( predictValue[i][k] == predictValue[j][k] )numOfAgree++;
					
					if(predictValue[i][k] == 0)a0++;
					if(predictValue[i][k] == 1)a1++; 
					if(predictValue[j][k] == 0)b0++;
					if(predictValue[j][k] == 1)b1++; 
				}
				
				phi_1 = (double)numOfAgree / (double)numInstances;
				phi_2 = (a0 / numInstances) * (b0 / numInstances) + (a1 / numInstances) * (b1 / numInstances);
				
			    double kappa = (phi_1 - phi_2) / (1 - phi_2);
			    kappa = (1 - kappa) / 2;
			    //System.out.println(kappa);
			    diversityScore += kappa;
			}
		}
		
		diversityScore = (diversityScore / count ) * 100;

		return diversityScore;
	}
	
	// double fault diversity
	public double doubleFaultDiversity(){
		double diversityScore = 0.0;
		double faultValue = 0.0;
		double overallValue = 0.0;
		
		for (int i = 0; i < numClassifiers; i++) {
			for (int j = i + 1; j < numClassifiers; j++) {
				int numOfFault = 0;
				faultValue = 0.0;
				
				for (int k = 0; k < numInstances; k++) {
					if (predictValue[i][k] == predictValue[j][k]) {
						if (givenValue[i] != predictValue[i][k]) numOfFault++;	
					}
				}
				
				faultValue = (double)numOfFault / (double)numInstances;
			    overallValue += faultValue;
			}
		}
		
		diversityScore = 1 - ((overallValue * 2) / (numClassifiers * (numClassifiers - 1)));
		diversityScore *= 100;
		//System.out.println("double fault: " + diversityScore);

		return diversityScore;
	}
	
	public double classify(Classifier c, int cId) throws Exception {
		
		// train the classifier with training data
		c.buildClassifier(train);
		
		// get the predict value and predict distribution from each test instances
		for (int i = 0; i < test.numInstances(); i++) {
			predictDistribution[cId][i] = c.distributionForInstance(test.instance(i));
			predictValue[cId][i] = c.classifyInstance(test.instance(i));
		}
		
		// of course, get the AUC for each classifier
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(c, test);
		return eval.areaUnderROC(1) * 100;
	}
	
	// this function calculate AUC for ensemble
	public double areaUnderROC(double[][] probList, int classIndex) {

		LinkedList<ClassSort> classSortLinkedList = new LinkedList<ClassSort>();
		for (int i = 0; i < probList.length; i++){
			int trueClass;
			if (classIndex == 0){
				if ((int)probList[i][1] == 0)
					trueClass = 1;
				else 
					trueClass = 0;
				}else
					trueClass = (int)probList[i][1];
			double prob = probList[i][0];
			//System.out.print(prob + " " + trueClass + "\n");
			classSortLinkedList.add(new ClassSort(prob, trueClass));
		}
		double roc = AUCCalculator.getROC(classSortLinkedList); 
		return roc;
	}
}