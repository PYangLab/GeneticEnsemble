/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    GeneticOperation.java
 *    Copyright (C) 2009 Pengyi Yang
 *    University of Sydney, Australia
 */
 
package au.edu.usyd.it.yangpy.snp;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.core.Instances;


/**
 * A parallel genetic optimization component
 * 
 * A parallel genetic optimization component which performing 
 * SNP subset evaluation with an ensemble of classifiers
 * through internal cross validation.
 * 
 * @author Pengyi Yang (yangpy7@gmail.com)
 * @version $Revision: 1.3
 */

public class ParallelGenetic {

	// number of thread for multiple threading
	private int numThread;
	
	// the balanced data set
	private Instances data;
	
	// data set of CV partition
	private Instances cvTrain;
	private Instances cvTest;
	
	// data set constrained according to the SNP set
	private Instances cTrain;
	private Instances cTest;
	
	/** population size */
	private int popSize;
	
	/** chromosome length */
	private int chroLen;
	
	/** termination generation */
	private int terGener;
	
	/** current generation index */
	private int currGener;
	
	/** crossover probability */
	private double pc;
	
	/** mutation probability */
	private double pm;
	
	/** tournament selection size */
	private int tourSize;
	
	/** a population of niched chromosomes */
	private int[][] chro;
	
	/** fitness of each chromosomes */
	private double[] chroFit;
	
	/** favored chromosomes */
	private int[][] chroFavored;
	
	private int elitismSize; // assume a default of 5
	private ArrayList<Integer> elitismList;
	
	/** execution mode, "v" for verbose, "s" for summarize */
	private String mode;
	
	/** diversity measure */
	private String diversity;
	
	/** cross validation fold index */
	private int foldSize;
	private int foldIndex;
	
	/** holds the best solution of genetic ensemble */
	private String bestChro;
	
	/** average fitness value of a generation */
	private double avgFitness;
	
	/** random number generator */
	private Random random = new Random(System.currentTimeMillis());
	
	
	/**
	 * Constructor.
	 * Initiate the user specified parameters.
	 * 
	 * @param fName		file name
	 * @param chroLen	GA chromosome size
	 * @param popSize	GA population size
	 * @param terGener	GA generation size
	 * @param fold		fold of internal CV
	 * @param mode		running mode
	 */
	public ParallelGenetic(Instances rawData, int chroLen, int popSize, int terGener, 
			String mode, boolean balance, String diversity, int numThread) {
		
		this.chroLen = chroLen;
		this.popSize = popSize;
		this.terGener = terGener;
		this.mode = mode;
		this.diversity = diversity;
		this.numThread = numThread;
		

		data = rawData;
		
		// apply sampling to balance original data if turned on
		if (balance == true) {
			
			// determine the numbers of cases and controls
			int cases = 0;
			int controls = 0;
			
			for (int i = 0; i < data.numInstances(); i++) {
				if (data.instance(i).classValue() == 1) {
					cases++;
				} else {
					controls++;
				}
			}
		
			// apply random over sampling to reduce class size difference
			if (controls > cases) {
				int dif = controls - cases;
				int count = 0;
				while (count < dif) {
					int i = random.nextInt(data.numInstances());
					if (data.instance(i).classValue() == 0) {
						data.delete(i);
						count++;
					}
				}
				
				System.out.println("original data: controls > cases; balanced");
			} else if (controls < cases) {
				int dif = cases - controls;
				int count = 0;
				while (count < dif) {
					int i = random.nextInt(data.numInstances());
					if (data.instance(i).classValue() == 1) {
						data.delete(i);
						count++;
					}
				}
				
				System.out.println("original data: controls < cases; balanced");
			} else {
				System.out.println("original data: controls = cases; no need to balance");
			}
		}
	}
	
	/**
	 * Initiate the internal parameters 
	 */
	public void initializeParameters () {

		// validate the given chromosome length with the SNP size
		if (data.numAttributes() - 1 < chroLen) {
			System.out.println();
			System.out.println(" The SNP size in the dataset is smaller than the defined chromosome length");
			System.out.println(" Using the SNP size as the chromosome length");
			chroLen = data.numAttributes() - 1;
		}
		
		// predefined parameters
		pc = 0.7;			// set crossover probability
		pm = 0.1;			// set mutation probability
		currGener = 0;		// set current generation index
		elitismSize = 5;	// set the size of elitisim
		elitismList = new ArrayList<Integer>();
		foldSize = 5;
		foldIndex = 0;
		
		// sample size related parameters
		/*
		if ((balanceData.numInstances()) > 2000 ) {
			if (fold == 0) fold = 50;
		} else if ((balanceData.numInstances()) > 1000 ) {
			if (fold == 0) fold = 20;
		} else if ((balanceData.numInstances()) > 500 ) {
			if (fold == 0) fold = 10;
		} else if ((balanceData.numInstances()) > 300 ) {
			if (fold == 0) fold = 5;
		} else if ((balanceData.numInstances()) > 100 ) { // 101-300
			if (fold == 0) fold = 3;
		} else { // 0 - 100
			if (fold == 0) fold = 2;
		}
		*/
		
		// data dimension related parameters
		if ((data.numAttributes() - 1) > 150) { // 151-
			if (chroLen == 0) chroLen = 22;
			if (popSize == 0) popSize = 400;
			if (terGener == 0) terGener = 40;
			tourSize = 7;
		} else if ((data.numAttributes() - 1) > 100) { // 101-150
			if (chroLen == 0) chroLen = 20;
			if (popSize == 0) popSize = 250;
			if (terGener == 0) terGener = 35;
			tourSize = 6;
		} else if ((data.numAttributes() - 1) > 50) { // 51-100
			if (chroLen == 0) chroLen = 18;
			if (popSize == 0) popSize = 250;
			if (terGener == 0) terGener = 45;
			tourSize = 5;
		} else if ((data.numAttributes() - 1) > 20) { // 21-50
			if (chroLen == 0) chroLen = 16;
			if (popSize == 0) popSize = 100;
			if (terGener == 0) terGener = 15;
			tourSize = 4;
		} else { // 0-20
			if (chroLen == 0) chroLen = 15;
			if (popSize == 0) popSize = 40;
			if (terGener == 0) terGener = 12;
			tourSize = 3;
		}
		
		chro = new int[popSize][chroLen];
		chroFavored = new int[popSize][chroLen];
		chroFit = new double[popSize];
		
		System.out.println("parameters: ");
		System.out.println(" - assign tournamet selection: " + tourSize);
		System.out.println(" - assign size of internal CV: " + foldSize);
		System.out.println(" - assign GA chromosome size: " + chroLen);
		System.out.println(" - assign GA generation size: " + terGener);
		System.out.println(" - assign GA population size: " + popSize);
		System.out.println("=============================================");
	}
	
	/**
	 * initialize genetic chromosomes
	 */
	/// checked on 18-08-2010
	public void initializeChromosomes () {
		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < chroLen; j++) {
				boolean isEmpty = false; // whether current position on a GA chromosome has a SNP
					
				// 1/2 odds that current locus on a GA chromosome may be empty.
				isEmpty = random.nextBoolean();
				
				if (isEmpty == true) {// -1 indicates that current locus does not contain a SNP
					chro[i][j] = -1;
				} else {
					while (true) {
						boolean contain = false;
						int sId = 0; // SNP index
						// generate a number between 0 and num_Of_SNPs
						sId = random.nextInt(data.numAttributes() - 1);
						
						for (int p = 0; p < j; p++) {
							if (chro[i][p] == sId) { // this SNP is already in the chromosome
								contain = true;
								break;
							}
						}
						
						if (contain == false) {
							chro[i][j] = sId;
							break;
						}
					}
				}
			}
		}
		
		// print initialized chromosomes
		System.out.println();
		System.out.println("initilization ");
		
		// update the cross validation training and testing sets
		crossValidate();
		
		System.out.println("---------------------------------------------");
		currGener++;
		System.out.println("generation: " + currGener + "\n");
		if(mode.equals("v")) printMatrix(chro, chroFit);
	}
	
	
	// Elitism selection
	public void selectElitism () {
		
		elitismList.clear();
		
		for (int i = 0; i < elitismSize; i++) {
			int eId = 0;
			double elitFit = Double.MIN_VALUE;
			
			for (int j = 0; j < popSize; j++) {
				
				if (elitFit < chroFit[j]) {
					if (elitismList.contains(j)) {
						continue;
					} else {
						elitFit = chroFit[j];
						eId = j;
					}
				} else if (elitFit == chroFit[j]) { // fit the fitness is the same, than favor the small size one
					
					if (elitismList.contains(j))
						continue;
					
					int jLen = 0;
					int eLen = 0;
					
					for (int k = 0; k < chroLen; k++) {
						if (chroFit[j] != -1) {
							jLen++;
						} 
						
						if (chroFit[eId] != -1) {
							eLen++;
						}
					}
					
					if (jLen < eLen) {
						eId = j;
					}
				}
			}
			
			elitismList.add(eId);
			//System.out.println("Elitist Id: " + eId);
			
			for (int k = 0; k < chroLen; k++) {
				chroFavored[i][k] = chro[eId][k];
			}
		}
	}
	
	
	/**
	 * tournament selection of chromosomes
	 */
	// fixed 18-08-2010
	public void selectUsingTournament () {

		int popCounter = 5; // reserve 0-4 for elitisim
		
		while (popCounter < popSize) {

			// randomly select the first chromosome
			int wId = random.nextInt(popSize);
			
			// select the favorite chromosome
			for (int k = 1; k < tourSize; k++) {
				
				int j = random.nextInt(popSize);
				while (wId == j) {
					j = random.nextInt(popSize);
				}
				
				if (chroFit[wId] < chroFit[j]) {
					wId = j;
				} else if (chroFit[wId] == chroFit[j]) {// favor the shorter chromosome if the fitness values are equal
					
					int wLen = 0;
					int jLen = 0;
					
					for (int m = 0; m < chroLen; m++) {
						if (chro[wId][m] != -1)
							wLen++;
						if (chro[j][m] != -1)
							jLen++;
					}

					if (wLen > jLen) {
						wId = j;
					}
				}
			}
			
			// copy the winner to the population pool
			for (int k = 0; k < chroLen; k++) {
				chroFavored[popCounter][k] = chro[wId][k];
			}
			
			popCounter++;
		}
	}
	
	/**
	 * implement genetic crossover
	 */
	// fixed on 18-08-2010
	public void crossover () {
		// chromosome A and B contain indicator
		boolean containA = false;
		boolean containB = false;
		
		// 0-4 are elitism which will be directly copy to next generation
		for (int j = 5; j < popSize - 1; j = j + 2) {
			if (random.nextDouble() < pc) {
				int cp = 0; // a crossover point
					
				// get a crossover point between 1 to (chroLen - 1)
				while ((cp = random.nextInt(chroLen)) == 0) 
					;
					 
				// conduct crossover
				int snpA, snpB;
					
				for (int p = 0; p < cp; p++) {
					snpA = chroFavored[j][p];
					snpB = chroFavored[j + 1][p];
					containA = false;
					containB = false;
						
					// check if the SNP is already in a given chromosome.
					for (int k = 0; k < chroLen; k++) { 
						if (snpB != -1)
							if (chroFavored[j][k] == snpB) 
								containA = true;
						if (snpA != -1)
							if (chroFavored[j + 1][k] == snpA) 
								containB = true;
					}
					
					// if it is not in current chromosome, put it in.
					if (containA == false)
						chroFavored[j][p] = snpB;
					if (containB == false)
						chroFavored[j + 1][p] = snpA;
				}
			}
		}
	}

	
	/**
	 * implement genetic mutation
	 */
	// fixed 18-08-2010
	public void mutate () {
		
		// 0-4 are elitism
		for (int j = 5; j < popSize; j++) {
			if (random.nextDouble() < pm) {
				int mp; // a mutate point
				int sId; // a holder of the changed value
				
				mp = random.nextInt(chroLen); // get a mutate point between 0 to (chroLen - 1)
				
				// 0.5 odds that current locus may be absent 
				boolean test = random.nextBoolean();
				
				if (test == true) {
					chroFavored[j][mp] = -1;
				} else {
					while (true) {
						// generate number between 0 and num_of_SNPs
						sId = random.nextInt(data.numAttributes() - 1);
						
						// check if the candidate SNP is already in this chromosome
						boolean contain = false;
						
						for (int k = 0; k < chroLen; k++) {
							if (chroFavored[j][k] == sId) {
								contain = true;
								break;
							}
						}
						
						// add in if it is not in current chromosome
						if (contain == false) {
							chroFavored[j][mp] = sId;
							break;
						}
					}
				}
			}
		}
	}
	
	
	public void generateNewGeneration () {
		// done all genetic processes, copy favored solutions to chromosome variable
		// and go to the next generation
		for(int i = 0; i < popSize; i++) {
			for(int j = 0; j < chroLen; j++) {
				chro[i][j] = chroFavored[i][j];
			}
		}

		// update the cross validation training and testing sets
		crossValidate();
		
		System.out.println("---------------------------------------------");
		currGener++;
		System.out.println("generation: " + currGener);
		System.out.println();
	}
	
	/**
	 * evaluate candidate solution
	 * 
	 * @param fId	the Id of the internal CV fold
	 */
	public void evaluate () {
				
		// initiate a thread pool
		ThreadPool threadPool = new ThreadPool(numThread);
		
		try {
			avgFitness = 0;
			
			// parallel the execution using multiple threading
			for (int i = 0; i < popSize; i++) {
				
				// for those without a single SNP, the fitness is 0
				int len = 0;
				for (int j = 0; j < chroLen; j++) {
					if (chro[i][j] != -1) {
						len++;
					}
				}
				
				if (len == 0) {
					chroFit[i] = 0.0;
				} else {
					threadPool.execute(new FitnessRunnable(i));
				}
			}
			threadPool.waitFinish();
			threadPool.closePool();
				
			// print the average fitness of a niche of population
			avgFitness /= popSize;
			DecimalFormat dec = new DecimalFormat("##.##");
			System.out.println("average fitness: " + dec.format(avgFitness));
			
			if(mode.equals("v")) {
				System.out.println("---------------- evaluation -----------------");
				printMatrix(chro, chroFit);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public class FitnessRunnable implements Runnable {
		private final int cId;
		
		public FitnessRunnable(int cId) {
			this.cId = cId;
		}

		public void run() {	
			// compute fitness for a chromosome
			try {
				chroFit[cId] = computeFitess(cId);
				//System.out.println(chroFit[id]);
			} catch (Exception ioe) {
				ioe.printStackTrace();
			}
		}
	}
	
	public void crossValidate () {
		// create a copy of original training set for CV
		Instances randData = new Instances(data);
		
		// divide the data set with x-fold stratify measure
		randData.stratify(foldSize);
		
		try {
		
			cvTrain = randData.trainCV(foldSize, foldIndex);
			cvTest = randData.testCV(foldSize, foldIndex);

			foldIndex++;
		
			if (foldIndex >= foldSize) {
				foldIndex = 0;
			}
		
		} catch (Exception e) {
			System.out.println(cvTest.toString());
		}
	}

	/**
	 * constrain data set with a given SNP subset
	 * 
	 * @param cId	chromosome Id
	 * @param train	training instances
	 * @param test	test instances
	 */
	public double computeFitess (int cId) throws Exception {

		Instances cTrain = new Instances(cvTrain);
		Instances cTest = new Instances(cvTest);
		
		int len = 0;
		for (int i = 0; i < chro[cId].length; i++) {
			if (chro[cId][i] != -1) {
				len++;
			}
		}
		
		int[] deleteList = new int[data.numAttributes() - 1 - len];
		
		int delId = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			
			boolean testContain = false;
			
			for (int j = 0; j < chro[cId].length; j++) {
				if (i == chro[cId][j]) {
					testContain = true;
				}
			}
			
			if (testContain == false) {
				deleteList[delId] = i;
				delId++;
			}
		}
		
		Arrays.sort(deleteList);
		// reverse the array 
		for (int i = 0; i < deleteList.length / 2; ++i) { 
			int temp = deleteList[i]; 
			deleteList[i] = deleteList[deleteList.length - i - 1]; 
			deleteList[deleteList.length - i - 1] = temp; 
		} 
		
		for (int i = 0; i < deleteList.length; i++) {
			cTrain.deleteAttributeAt(deleteList[i]);
			cTest.deleteAttributeAt(deleteList[i]);
		}
		
		
		////////////////////////////////////////////
		// compute fitness
		double fitness = 0.0;
		
		//boolean useDiversity = false;
		
		if (mode.equals("v")) {
			System.out.println("---------------------------------------------");
			System.out.println(" subset " + (cId + 1) + ":");
			System.out.println();
		}
		
		Ensemble classifier = new Ensemble(cTrain, cTest);
		classifier.ensemble(mode);
		double blockScore = classifier.blocking();
		double voteScore = classifier.voting();
		double diversityScore = 0.0;
		
		if (currGener < (terGener - (terGener / 5))) {
			if (diversity.equals("K")) {
				diversityScore = classifier.kappaDiversity();
			} else {
				diversityScore = classifier.doubleFaultDiversity();
			}
		}
		
		blockScore = Math.round(blockScore * 10000.0) / 10000.0;
		voteScore = Math.round(voteScore * 10000.0) / 10000.0;
		
		if (diversityScore != 0.0) {
			diversityScore = Math.round(diversityScore * 10000.0) / 10000.0;
			fitness = blockScore * 0.45 + voteScore * 0.45 + diversityScore * 0.1;
		} else {
			fitness = blockScore * 0.5 + voteScore * 0.5;
		}
		
		// average accuracy of five classifiers.
		if(mode.equals("v")){
			System.out.println("block (average) AUC: " + blockScore);
			System.out.println("majority voting AUC: " + voteScore);
			
			if (diversityScore != 0.0) {
				if (diversity.equals("K")) {
					System.out.println("kappa diversity: " + diversityScore);
				} else {
					System.out.println("double fault diversity: " + diversityScore);
				}
			}
		}
		
		avgFitness += fitness ;
		return fitness;
		
	}
	



	
	/**
	 * save the best chromosome (subset)
	 * 
	 * @param append	append or overwrite the file
	 */
	public void saveBestChro (boolean append) {

		// find the best chromosome
		bestChro = findBestChromosome();
		
		try {
			
			if (append == false) { 
				// write to the file
				BufferedWriter bw = new BufferedWriter(new FileWriter("snpSet.out"));
				String[] snpSet = bestChro.split("\\t");
				
				for (int i = 0; i < snpSet.length - 1; i++) {
					String[] fId = data.attribute(Integer.parseInt(snpSet[i])).toString().split("\\s+");
					bw.write(fId[1] + "\t");
				}
				
				String[] fId = data.attribute(Integer.parseInt(snpSet[snpSet.length - 1])).toString().split("\\s+");
				bw.write(fId[1]);
				bw.newLine();
				bw.close();
			} else {	
				// append to the file
				BufferedWriter bw = new BufferedWriter(new FileWriter("snpSet.out", true));
				String[] snpSet = bestChro.split("\\t");
				
				for (int i = 0; i < snpSet.length - 1; i++) {
					String[] fId = data.attribute(Integer.parseInt(snpSet[i])).toString().split("\\s+");
					bw.append(fId[1] + "\t");
				}
				
				String[] fId = data.attribute(Integer.parseInt(snpSet[snpSet.length - 1])).toString().split("\\s+");
				bw.append(fId[1]);
				bw.newLine();
				bw.close();
			}
			
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * find the best subset through different niches
	 * 
	 * @return	best chromosome (subset)
	 */
	public String findBestChromosome () {
		String bestChro = "";
		double maxFit = 0.0;
		int minSize = Integer.MAX_VALUE;
		int cId = 0; // chromosome Id
		
		// find the maximum fitness
		for (int i = 0; i < popSize; i++) {
			if (maxFit < chroFit[i]) {
				maxFit = chroFit[i];
			}
		}
		
		// find the chromosome with maximum fitness and smallest size
		for (int i = 0; i < popSize; i++) {
			if (maxFit == chroFit[i]) {
				int len = 0;
				for (int j = 0; j < chroLen; j++) {
					if (chro[i][j] != -1) 
						len++;
				}
				
				if (minSize > len) {
					minSize = len;
					cId = i;
				}
			}
		}


		// copy this chromosome as the best SNP combination
		for (int i = 0; i < chroLen; i++) {
			// if no SNP in current locus, continue the loop.
			if(chro[cId][i] == -1) {
				continue;
			}
			
			Integer piece = new Integer(chro[cId][i]);
			bestChro = bestChro + piece + "\t";
		}
		
		if(mode.equals("v")) {
			System.out.println("final generation: ");
			printMatrix(chro, chroFit);
		}
		
		System.out.println("=============================================");
		System.out.println("best subset in codeing form:");
		System.out.println(bestChro);
		System.out.println("fitness of the fubset: " + maxFit);
		
		return bestChro;
	}

	/**
	 * print chromosome of every niche with its fitness value
	 * 
	 * @param m1	chromosome matrix
	 * @param m2	chromosome fitness value matrix
	 */
	public void printMatrix(int[][] m1, double[]m2) {
		
		// print title
		System.out.print("subset\t");
		for (int j = 0; j < chroLen; j++) {
			System.out.print("f_" + j + "\t");
		}
		System.out.println("fitness");

		
		for (int i = 0; i < m1.length; i++) {
			System.out.print("set_" + i + "\t");
			
			for (int j = 0; j < m1[i].length; j++) {
				System.out.print(m1[i][j] + "\t");
			}
			
			System.out.println(m2[i]);
		}
		
		System.out.println();
	}
	
	/**
	 * get current generation
	 * 
	 * @return current generation
	 */
	public int getCurrentGeneration () {
		return currGener;
	}
	
	/**
	 * get termination generation
	 * 
	 * @return termination generation
	 */
	public int getTerimateGeneration () {
		return terGener;
	}
}
