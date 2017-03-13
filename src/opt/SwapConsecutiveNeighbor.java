package opt;

import shared.Instance;
import dist.Distribution;

public class SwapConsecutiveNeighbor implements NeighborFunction {
    
    /**
     * @see opt.ga.MutationFunction#mutate(opt.OptimizationData)
     */
    public Instance neighbor(Instance d) {
        Instance cod = (Instance) d.copy();
        int i = Distribution.random.nextInt(cod.getData().size());
        int j = (i == cod.getData().size() - 1) ? 0 : i + 1;
        double temp = cod.getContinuous(i);
        cod.getData().set(i, cod.getContinuous(j));
        cod.getData().set(j, temp);
        return cod;
    }
}