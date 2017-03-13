import java.awt.Point;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapConsecutiveNeighbor;
import opt.SwapNeighbor;
import opt.example.CountOnesEvaluationFunction;
import opt.example.FlipFlopEvaluationFunction;
import opt.example.FourPeaksEvaluationFunction;
import opt.example.KnapsackEvaluationFunction;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * modified by Yeshwant Dattatreya
 * @version 1.0
 */

class Analyze_Optimization_Test implements Runnable {

    private Thread t;

    private String problem;
    private String algorithm;
    private int iterations;
    private HashMap<String, Double> params;
    private int N;
    private int T;
    private ConcurrentHashMap<String, String> other_params;
    private int run;
    private double[][] points;
    private HashMap<String, Object> problem_params;
    

    Analyze_Optimization_Test(
            String problem,
            String algorithm,
            int iterations,
            HashMap<String, Double> params,
            int N,
            int T,
            ConcurrentHashMap<String, String> other_params,
            int run,
            double[][] points,
            HashMap<String, Object> problem_params
        ) {
        this.problem = problem;
        this.algorithm = algorithm;
        this.iterations = iterations;
        this.params = params;
        this.N = N;
        this.T = T;
        this.other_params = other_params;
        this.run = run;
        this.points = points;
        this.problem_params = problem_params;
    }

    private void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void run() {
        try {
            EvaluationFunction ef = null;
            Distribution odd = null;
            NeighborFunction nf = null;
            MutationFunction mf = null;
            CrossoverFunction cf = null;
            Distribution df = null;
            
            ProbabilisticOptimizationProblem pop = null;
            boolean runTillConvergence = false;
            double globalOptimimumVal = -1;
            
            if (this.iterations == -1) {
            	runTillConvergence = true;
            }
            
            int[] ranges;
            switch (this.problem) {
                case "count_ones":
                    ranges = new int[this.N];
                    Arrays.fill(ranges, 2);
                    ef = new CountOnesEvaluationFunction();
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new UniformCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
                    break;
                case "four_peaks":
                    ranges = new int[this.N];
                    Arrays.fill(ranges, 2);
                    ef = new FourPeaksEvaluationFunction(this.T);
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new SingleCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
                    globalOptimimumVal = ef.getGlobalMax(this.N);
                    //System.out.println("global: " + globalOptimimumVal);
                    break;
                case "tsp":
                    ef = new TravelingSalesmanRouteEvaluationFunction(points);
                    odd = new DiscretePermutationDistribution(N);
                    nf = new SwapNeighbor();
                    //nf = new SwapConsecutiveNeighbor();
                    mf = new SwapMutation(); // Does the same thing
                    cf = new TravelingSalesmanCrossOver((TravelingSalesmanEvaluationFunction) ef);
                    ranges = new int[N];
                    Arrays.fill(ranges, N);
                    df = new DiscreteDependencyTree(.1, ranges);
                    pop = new GenericProbabilisticOptimizationProblem(new TravelingSalesmanSortEvaluationFunction(points), new  DiscreteUniformDistribution(ranges), df);
                    break;
                case "knapsack":
                	
                    double[] weights = (double[])problem_params.get("weights");
                    double[] volumes = (double[])problem_params.get("volumes");
                    int copiesEach = (int)problem_params.get("COPIES_EACH");
                    double knapSackVolume = (double)problem_params.get("KNAPSACK_VOLUME");
                    
                    int[] copies = new int[this.N];
                    Arrays.fill(copies, copiesEach);
                    
                    ranges = new int[this.N];
                    Arrays.fill(ranges, copiesEach + 1);
                    
                    ef = new KnapsackEvaluationFunction(weights, volumes, knapSackVolume, copies);
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new UniformCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges); 

                    pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                    break;
                case "flipflop":
                	ranges = new int[N];
                    Arrays.fill(ranges, 2);
                    ef = new FlipFlopEvaluationFunction();
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new SingleCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
            }
            
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            if (pop == null) {
            	pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            }

            StringBuilder results = new StringBuilder("");
            double optimal_value = -1;
            double function_evaluations = 0;

            int convergenceNumber = 1000;
            switch (this.algorithm) {
                case "RHC":
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    if (runTillConvergence) {
                    	if (globalOptimimumVal > 0) {
                    		double currentVal = 0;
                    		double prevVal = 0;
                    		int sameValCount = 0;
                    		do {
                    			currentVal = rhc.train();
                    			results.append(getPrintingValue(this.problem, currentVal) + "\n");
                    			if (prevVal == currentVal) {
                    				sameValCount++;
                    			} else {
                    				sameValCount = 0;
                    			}
                    			prevVal = currentVal;
                    		} while ((int)globalOptimimumVal != (int)currentVal && sameValCount < convergenceNumber);
                    	}
                    } else {
                    	for (int i = 0; i <= this.iterations; i++) {
                            results.append(getPrintingValue(this.problem, rhc.train()) + "\n");
                        }
                    }
                    
                    optimal_value = getPrintingValue(this.problem, ef.value(rhc.getOptimal()));
                    function_evaluations = rhc.getFunctionEvaluations();
                    break;

                case "SA":
                    SimulatedAnnealing sa = new SimulatedAnnealing(
                            params.get("SA_initial_temperature"),
                            params.get("SA_cooling_factor"),
                            hcp
                    );
                    
                    if (runTillConvergence) {
                    	if (globalOptimimumVal > 0) {
                    		double currentVal = 0;
                    		double prevVal = 0;
                    		int sameValCount = 0;
                    		do {
                    			currentVal = sa.train();
                    			results.append(getPrintingValue(this.problem, currentVal) + "\n");
                    			if (prevVal == currentVal) {
                    				sameValCount++;
                    			} else {
                    				sameValCount = 0;
                    			}
                    			prevVal = currentVal;
                    		} while ((int)globalOptimimumVal != (int)currentVal && sameValCount < convergenceNumber);
                    	}
                    } else {
                    	for (int i = 0; i <= this.iterations; i++) {
                            results.append(getPrintingValue(this.problem, sa.train()) + "\n");
                        }
                    }
                    
                    optimal_value = getPrintingValue(this.problem, ef.value(sa.getOptimal()));
                    function_evaluations = sa.getFunctionEvaluations();
                    break;

                case "GA":
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(
                            params.get("GA_population").intValue(),
                            params.get("GA_mate_number").intValue(),
                            params.get("GA_mutate_number").intValue(),
                            gap
                    );
                    
                    if (runTillConvergence) {
                    	if (globalOptimimumVal > 0) {
                    		double currentVal = 0;
                    		double prevVal = 0;
                    		int sameValCount = 0;
                    		do {
                    			currentVal = ga.train();
                    			results.append(getPrintingValue(this.problem, currentVal) + "\n");
                    			if (prevVal == currentVal) {
                    				sameValCount++;
                    			} else {
                    				sameValCount = 0;
                    			}
                    			prevVal = currentVal;
                    		} while ((int)globalOptimimumVal != (int)currentVal && sameValCount < convergenceNumber);
                    	}
                    } else {
                    	for (int i = 0; i <= this.iterations; i++) {
                            results.append(getPrintingValue(this.problem, ga.train()) + "\n");
                        }
                    }
                    
                    optimal_value = getPrintingValue(this.problem, ef.value(ga.getOptimal()));
                    function_evaluations = ga.getFunctionEvaluations();
                    break;

                case "MIMIC":
                    MIMIC mimic = new MIMIC(
                            params.get("MIMIC_samples").intValue(),
                            params.get("MIMIC_to_keep").intValue(),
                            pop
                    );
                    //results = "";
                    if (runTillConvergence) {
                    	if (globalOptimimumVal > 0) {
                    		double currentVal = 0;
                    		double prevVal = 0;
                    		int sameValCount = 0;
                    		do {
                    			currentVal = mimic.train();
                    			results.append(getPrintingValue(this.problem, currentVal) + "\n");
                    			if (prevVal == currentVal) {
                    				sameValCount++;
                    			} else {
                    				sameValCount = 0;
                    			}
                    			prevVal = currentVal;
                    		} while ((int)globalOptimimumVal != (int)currentVal && sameValCount < convergenceNumber);
                    	}
                    } else {
                    	double prevVal = 0;
                    	for (int i = 0; i <= this.iterations || (globalOptimimumVal > 0 && globalOptimimumVal == prevVal); i++) {
                            results.append(getPrintingValue(this.problem, mimic.train()) + "\n");
                        }
                    }

                    optimal_value = getPrintingValue(this.problem, ef.value(mimic.getOptimal()));
                    function_evaluations = mimic.getFunctionEvaluations();
                    break;
            }
            results.append("\n" +
                    "Problem: " + this.problem + "\n" +
                    "Algorithm: " + this.algorithm + "\n" +
                    "Optimal Value: " + optimal_value + "\n" +
                    "Fitness function evaluations: " + function_evaluations + "\n");
            String final_result = "";
            final_result =
                    this.problem + "," +
                    this.algorithm + "," +
                    this.N + "," +
                    this.iterations + "," +
                    this.run + "," +
                    optimal_value + "," +
                    function_evaluations;
            write_output_to_file(this.other_params.get("output_folder"), "final_results.csv", final_result, true);
            String file_name =
                    this.problem + "_" + this.algorithm + "_N_" + this.N +
                    "_iter_" + this.iterations + "_run_" + this.run + ".csv";
            write_output_to_file(this.other_params.get("output_folder"), file_name, results.toString(), false);
            System.out.println(results);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public double getPrintingValue(String problem, double optimalValue) {
    	switch(problem) {
    		case "tsp": return 1/optimalValue;
    	}
    	return optimalValue;
    }

    public void start () {
        if (t == null)
        {
            t = new Thread (this);
            t.start ();
        }
    }
    
    public Pair meanAndSDE(double [] array) {
    	double avg = 0, std = 0;
    	for (double element : array) {
    		avg += element;
    	}
    	
    	avg /= array.length;
    	
    	for(double element : array){
    		std += Math.pow((element - avg), 2);
    	}
    	std = Math.sqrt(std);
    	
    	Pair p = new Pair(avg, std);
    	return p;
    }
}

class Pair {
	double l, r;
	public Pair(double l, double r) {
		this.l = l;
		this.r = r;
	}
}

public class OptimizationTest {
	/** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
    
    public static void main(String[] args) {

        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results2");
        //int num_runs = 3;

        //Count Ones Test
        /*HashMap<String, Double> count_one_test_params = new HashMap<>();
        count_one_test_params.put("SA_initial_temperature",10.);
        count_one_test_params.put("SA_cooling_factor",.95);
        count_one_test_params.put("GA_population",20.);
        count_one_test_params.put("GA_mate_number",20.);
        count_one_test_params.put("GA_mutate_number",5.);
        count_one_test_params.put("MIMIC_samples",50.);
        count_one_test_params.put("MIMIC_to_keep",10.);

        int[] N = {10,20};
        int[] iterations = {10,20,30};
        String[] algorithms = {"RHC", "SA", "GA", "MIMIC"};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                //count_one_test_params.put("N",(double)N[j]);
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        //other_params.remove("run");
                        //other_params.put("run","" + l);
                        new Analyze_Optimization_Test(
                                "count_ones",
                                algorithms[i],
                                iterations[k],
                                count_one_test_params,
                                N[j],
                                0, //this doesn't apply to count ones problem, so simnply pass a 0
                                other_params,
                                l,
                                null,
                                null
                        ).start();
                    }
                }
            }
        }*/

        int num_runs = 10;
        String[] algorithms = {"RHC", "SA", "GA", "MIMIC"};
        //String[] algorithms = {"SA"};
        //int[] N = new int[] {40, 50, 60, 70, 80, 90, 100};
        int[] N = new int[] {200};
        int[] iterations = {20000};
        //int[] iterations = {-1};
        
        //Four Peaks Test
        HashMap<String, Double> four_peaks_test_params = new HashMap<>();
        four_peaks_test_params.put("SA_initial_temperature",1E11);
        four_peaks_test_params.put("SA_cooling_factor",.95);
        four_peaks_test_params.put("GA_population",200.);
        four_peaks_test_params.put("GA_mate_number",100.);
        four_peaks_test_params.put("GA_mutate_number",10.);
        four_peaks_test_params.put("MIMIC_samples",200.);
        four_peaks_test_params.put("MIMIC_to_keep",20.);

        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                //four_peaks_test_params.put("N",(double)N[j]);
                //four_peaks_test_params.put("T", (double)N[j]/5);
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        //other_params.remove("run");
                        //other_params.put("run", "" + l);
                        new Analyze_Optimization_Test(
                                "four_peaks",
                                algorithms[i],
                                algorithms[i].equals("MIMIC") ? 2500 : iterations[k],
                                four_peaks_test_params,
                                N[j],
                                N[j] / 5,
                                other_params,
                                l,
                                null,
                                null
                        ).start();
                    }
                }
            }
        }
        
        /*int num_runs = 1;
        
        HashMap<String, Double> tsp_test_params = new HashMap<>();
        tsp_test_params.put("SA_initial_temperature",1E12);
        tsp_test_params.put("SA_cooling_factor",0.95);
        tsp_test_params.put("GA_population",200.);
        tsp_test_params.put("GA_mate_number",100.);
        tsp_test_params.put("GA_mutate_number",10.);
        tsp_test_params.put("MIMIC_samples",200.);
        tsp_test_params.put("MIMIC_to_keep",20.);
        tsp_test_params.put("MIMIC_to_keep",20.);
        
        Random random = new Random();
        double[][] points = new double[200][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        
        String[] algorithms = {"RHC", "SA"};
        //String[] algorithms = {"RHC"};
        int[] N = new int[] {50};
        int[] iterations = {2000};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        //other_params.remove("run");
                        //other_params.put("run", "" + l);
                        new Analyze_Optimization_Test(
                                "tsp",
                                algorithms[i],
                                iterations[k],
                                tsp_test_params,
                                N[j],
                                0, // does not apply to TSP
                                other_params,
                                l,
                                points
                        ).start();
                    }
                }
            }
        }*/
        
        /*double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
        
        HashMap<String, Object> knapsackParams = new HashMap<String, Object>();
        knapsackParams.put("weights", weights);
        knapsackParams.put("volumes", volumes);
        knapsackParams.put("KNAPSACK_VOLUME", KNAPSACK_VOLUME);
        knapsackParams.put("COPIES_EACH", COPIES_EACH);*/
        
        //TODO: n = NUM_ITEMS
        
        /*int num_runs = 10;
        String[] algorithms = {"RHC", "SA", "GA", "MIMIC"};
        //String[] algorithms = {"RHC"};
        int[] N = new int[] {NUM_ITEMS};
        int[] iterations = {20000};
        //int[] iterations = {-1};
        
        //Four Peaks Test
        HashMap<String, Double> knapsack_test_params = new HashMap<>();
        knapsack_test_params.put("SA_initial_temperature",1E2);
        knapsack_test_params.put("SA_cooling_factor",.995);
        knapsack_test_params.put("GA_population",200.);
        knapsack_test_params.put("GA_mate_number",100.);
        knapsack_test_params.put("GA_mutate_number",10.);
        knapsack_test_params.put("MIMIC_samples",200.);
        knapsack_test_params.put("MIMIC_to_keep",20.);

        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        new Analyze_Optimization_Test(
                                "knapsack",
                                algorithms[i],
                                algorithms[i].equals("MIMIC") ? 3000 : iterations[k],
                                knapsack_test_params,
                                N[j],
                                N[j] / 5,
                                other_params,
                                l,
                                null,
                                knapsackParams
                        ).start();
                    }
                }
            }
        }*/
        
        
        /*int num_runs = 1;
        String[] algorithms = {"RHC", "SA", "GA", "MIMIC"};
        int[] N = new int[] {80};
        int[] iterations = {20000};
        //int[] iterations = {-1};
        
        //Flipflop Peaks Test
        HashMap<String, Double> flipflop_test_params = new HashMap<>();
        flipflop_test_params.put("SA_initial_temperature",1E2);
        flipflop_test_params.put("SA_cooling_factor",.95);
        flipflop_test_params.put("GA_population",200.);
        flipflop_test_params.put("GA_mate_number",100.);
        flipflop_test_params.put("GA_mutate_number",10.);
        flipflop_test_params.put("MIMIC_samples",200.);
        flipflop_test_params.put("MIMIC_to_keep",20.);

        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        new Analyze_Optimization_Test(
                                "flipflop",
                                algorithms[i],
                                iterations[k],
                                //algorithms[i].equals("MIMIC") ? 20000 : iterations[k],
                                flipflop_test_params,
                                N[j],
                                N[j] / 5,
                                other_params,
                                l,
                                null,
                                knapsackParams
                        ).start();
                    }
                }
            }
        }*/
    }
}