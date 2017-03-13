import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.Arrays;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksOpt {
    /** The n value */
    private static final int N = 50;
    /** The t value */
    private static final int T = N / 5;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        for (int iter = 1; iter < 101; iter++) {

                double start = System.nanoTime(), end, trainingTime;
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200 * iter);
                fit.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                System.out.println(trainingTime);
                System.out.println( ef.value(rhc.getOptimal()));

                start = System.nanoTime();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .97, hcp);
                fit = new FixedIterationTrainer(sa, 200 * iter);
                fit.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                System.out.println(trainingTime);
                System.out.println( ef.value(sa.getOptimal()));

                start = System.nanoTime();

                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(50, 30, 20, gap);
                fit = new FixedIterationTrainer(ga, 200 * iter);
                fit.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                System.out.println(trainingTime);
                System.out.println( ef.value(ga.getOptimal()));

                start = System.nanoTime();

                MIMIC mimic = new MIMIC(200, 20, pop);
                fit = new FixedIterationTrainer(mimic, 20 * iter);
                fit.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                System.out.println(trainingTime);
                System.out.println( ef.value(mimic.getOptimal()));

            }

    }

    public static double average(double[] result_list) {
        // 'average' is undefined if there are no elements in the list.
        // Calculate the summation of the elements in the list
        long sum = 0;
        int n = result_list.length;
        // Iterating manually is faster than using an enhanced for loop.
        for (int i = 0; i < n; i++)
            sum += result_list[i];
        // We don't want to perform an integer division, so the cast is mandatory.
        return ((double) sum) / n;
    }

    public static int endlocal(double[] result_list,int local){
        int time = 0;
        int n = result_list.length;
        for (int i = 0; i < n; i++){
            if (result_list[i]<=local){
                time +=1;
            }
        }
        return time;
    }

    public static int endglobal(double[] result_list,int global){
        int time = 0;
        int n = result_list.length;
        double d = global*1.8-1;
        for (int i = 0; i < n; i++){
            if (result_list[i]==d){
                time +=1;
            }
        }
        return time;
    }

}


