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
 *    GEsnpxPara.java
 *    Copyright (C) 2011-2015 Pengyi Yang
 *    The University of Sydney, Australia
 */


package au.edu.usyd.it.yangpy.snp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;

import weka.core.Instances;


/**
 *	General Description:
 *		GEsnpxPara system. An parallel implementation of genetic ensemble algorithm for 
 *  gene-gene interaction identification.
 *  
 *	<ol>
 *	Ensemble of classifiers: 
 *		<li> J48
 *		<li> CT1NN
 *		<li> CT3NN
 *      <li> KStar
 *		<li> CT5NN
 *	</ol>
 * 
 *
 * @author Pengyi Yang (yangpy@it.usyd.edu.au)
 * @version $Revision: 1.2
 * 
 * ----------------
 * Creation Date: 
 * 01/05/2007
 * 
 * Modification Date:
 * 10/09/2008
 * 14/05/2009
 * 24/09/2009
 * 03/12/2009
 * 03/09/2010
 * 09/11/2010
 * 12/13/2011
 */

public class GEsnpxPara {
	
	/** initiate thread */
	private int numThread = 1;
	
	/** name of the input data file */
	private String file = null;
	
	/** meta iteration time of genetic ensemble (default is 25) */
	private int iteration = 20;
	
	/** genetic parameters */
	private int chroLen;
	private int popSize;
	private int terGener;
	
	/** execution mode (default is summary) */
	private String mode = "s";
	
	/** apply sampling to balance data (default is false)*/
	private boolean balance = false;
	
	/** diversity measure (default is kappa) */
	private String diversity = "K";
	
	/**
	 * main function of GEsnpx
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		GEsnpxPara ge = new GEsnpxPara();
		
		// show usage
		if (args.length == 0) {
			usage();
		} else {
			for (int i = 0; i < args.length; i++) {
				if (args[i].equalsIgnoreCase("-h") || 
						args[i].equalsIgnoreCase("h") || 
						args[i].equalsIgnoreCase("-help") || 
						args[i].equalsIgnoreCase("help")) {
						usage();
				} if (args[i].equals("-f")) {
					ge.file = args[i+1];
					if (ge.file == null) {
						usage();
					}	
				}
			}
		}

		// genetic parameters (optional)
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-i")) {
				ge.iteration = Integer.parseInt(args[i+1]);
			} if (args[i].equals("-d")) {
				ge.chroLen = Integer.parseInt(args[i+1]);
			} if (args[i].equals("-p")) {
				ge.popSize = Integer.parseInt(args[i+1]);
			} if (args[i].equals("-g")) {
				ge.terGener = Integer.parseInt(args[i+1]);
			} if (args[i].equals("-v")) {
				ge.mode = "v";
			} if (args[i].equals("-s")) {
				ge.mode = "s";
			} if (args[i].equals("-b")) {
				ge.balance = true;
			} if (args[i].equals("-x")) {
				ge.diversity = args[i+1];
			} if (args[i].equals("-t")) {
				ge.numThread = Integer.parseInt(args[i+1]);
			}
		}
				
		// timing the procedure
		long t0 = System.currentTimeMillis();
		
		// perform genetic operations
		for (int i = 0; i < ge.iteration; i++) {
			System.out.println();
			System.out.println("=============================================");
			System.out.println("iteration: " + (i + 1));
			ge.performGeneticOperation(i);
		}

		// perform combinatorial ranking
		ge.performCombinatorialRanking();
		
		// stop timing and print the time spent
		long t1 = System.currentTimeMillis();
		long runTime = t1 - t0;
		
		// to seconds
		runTime /= 1000;
		// to minutes
		long sec = runTime % 60;
		long min = runTime / 60;
		
		System.out.println();
		System.out.println("==================================================");
		System.out.println("time spent = " + min + " mins " + sec + " secs");
		System.out.println("interaction identification process accomplished.");
		System.out.println("candidate interactions are written in 'interaction.rank'");
	}
	
	
	/**
	 * this function perform genetic operations
	 *
	 * @param saveFlag	append/write to the output file
	 */
	public void performGeneticOperation (int saveFlag) throws Exception {
		// initialize processing components
		// loading the raw data
		Instances rawData = new Instances(new BufferedReader(new FileReader(file)));
		rawData.setClassIndex(rawData.numAttributes() - 1);
		
		ParallelGenetic genetic = new ParallelGenetic(rawData ,chroLen ,popSize ,terGener, mode, balance, diversity, numThread);
		genetic.initializeParameters();
		genetic.initializeChromosomes();
		genetic.evaluate();

		for (int i = 1; i < genetic.getTerimateGeneration(); i++) {
			genetic.selectElitism();
			genetic.selectUsingTournament();
			genetic.crossover();
			genetic.mutate();
			genetic.generateNewGeneration();
			genetic.evaluate();
		}
		
		if (saveFlag == 0)
			genetic.saveBestChro(false);
		else
			genetic.saveBestChro(true);
	}
	
	////////////
	public void performCombinatorialRanking() throws Exception {
		Hashtable<String, Integer>  snpCombi = new Hashtable<String, Integer>();
		
		BufferedReader br = new BufferedReader(new FileReader("snpSet.out"));
		String line = "";
		
		while ((line = br.readLine()) != null) {
			String[] snps = line.split("\\t");
			Arrays.sort(snps);
			int[] indices;
			
			for (int k = 2; k <= snps.length; k++) {
				CombinationGenerator x = new CombinationGenerator (snps.length, k);
				StringBuffer combination;
				
				while (x.hasMore()) {
					
					combination = new StringBuffer();
					indices = x.getNext();
					
					for (int i = 0; i < indices.length; i++) {
						if (i == 0) {
							combination.append(snps[indices[i]]);
						} else {
							combination.append("-" + snps[indices[i]]);
						}
					}
					
					// record the SNP combinations
					if (snpCombi.containsKey(combination.toString())) {
						Integer C = (Integer)snpCombi.get(combination.toString());
						int c = C.intValue();
						c++;
						C = new Integer(c);
						snpCombi.put(combination.toString(), C);
					} else {
						snpCombi.put(combination.toString(), 1);
					}
				}
			}
		}
		br.close();
		
		// Sort the hash table
		BufferedWriter bw = new BufferedWriter(new FileWriter("interaction.rank"));
		
		System.out.println();
		System.out.println("==============  Combinatorial Ranking  =============");
		bw.write("==============  Combinatorial Ranking  =============");
		bw.newLine();
		int count = 2;
		
		while (count <= iteration) {
			Integer ccurCount = new Integer(count);
			Iterator<String> itr;
			itr = snpCombi.keySet().iterator();		
			
			while (itr.hasNext()) {
				String key = (String)itr.next();
				String value = snpCombi.get(key).toString();
				if (value.equals(ccurCount.toString())) {
					// calculate the identification frequency
					double freq = Double.parseDouble(value) / (double)iteration;
					System.out.println(key + "\t" + freq);
					bw.write(key + "\t" + freq);
					bw.newLine();
				}
			}
			count++;
		}
		bw.close();
	}
	
	public static void usage () {
		
		System.out.println("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"+"*\n"+
							"*                              Welcome to GEsnpxPara \n"+
							"* (A parallel implemntation of genetic ensemble system for gene-gene detection) \n*\n"+
							"* Copyright (C) 2011-2015; Pengyi Yang\n"+
							"* Email:\tyangpy7@gmail.com\n"+
							"* Institute:\tSchool of IT, University of Sydney, Australia\n"+
							"* Update:\t13 Dec. 2011; Version: 1.2\n*");

		System.out.println("===========================================================================\n"+"*\n"+
							"* General description:\n"+ 
							"*   GEsnpxPara is a parallel implementation of a genetic ensemble algorithm \n" +
							"* developed for gene-gene interaction identification. The system utilizes \n" +
							"* a multiple objective genetic algorithm with an ensemble of 5 nonlinear \n"+ 
							"* classifiers to capture gene-gene interactions through SNP markers. SNP \n" +
							"* subsets are evaluated and selected in a combinatorial manner, and potential\n"+
							"* interactions are identified by a combinatorial ranking procedure.\n"+
				   			"*\n===========================================================================\n*");
	
		System.out.println("* Usage:\n"+
							"* \tjava -jar GEsnpxPara.jar -f <dataset> [options]\n" +
							"* \t\t[OR] \n" +
							"* \tjava au.edu.usyd.it.yangpy.snp.GEsnpxPara -f <dataset> [options]\n*");

		System.out.println("* Dataset:\n"+
				   			"* \t<dataset>\t->\tdata file should be a matrix in ARFF format\n"+
				   			"* \t\t\t\t(see ARFF format for more details)");

		System.out.println("* General options:\n"+
				   			"* \t-h \t\t->\tprint this help\n"+
				   			"* \t-f <dataset>\t->\tspecify the data file\n"+
				   			"* \t-t <int>\t->\tnumber of threads (default=1)\n"+
							"* \t-v \t\t->\tverbose mode\n"+
				   			"* \t-s \t\t->\tsummary mode\n"+
				   			"* \t-b \t\t->\tbalance the data (for highly imbalanced data)\n"+
				   			"* \t-x <K|D>\t->\tdiversity (K = kappa; D = double fault)\n*");
				   			
		System.out.println("* Genetic algorithm parameters:\n"+
							"* (The default values of these parameters are assigned according to\n" +
							"*  the sample size and SNP size. You can specify alternative values.)\n"+
							"* \t-i <int>\t->\tnumber of GEsnpPara iteration\n"+
							"* \t-d <int>\t->\tsize of GA chromosome\n"+
							"* \t-p <int>\t->\tsize of GA population\n"+
							"* \t-g <int>\t->\tnumber of GA generation\n*");
		
		System.out.println("* Reference:\n"+
							"*   Pengyi Yang, Joshua WK Ho, Albert Y Zomaya, Bing B Zhou. \n"+
							"*   A genetic ensemble approach for gene-gene interaction identification.\n"+ 
							"*   BMC Bioinformatics 2010, 11:524.");
		
		System.out.println("* \n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
		
		System.exit(1);
	}
}

