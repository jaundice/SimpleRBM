using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface ILearningRateCalculator<T>
    {
        T CalculateLearningRate(int layer, int epoch);
    }


    public interface ILearningRateCalculatorFactory<T>
    {
        ILearningRateCalculator<T> Create(int layer);
    }
}
