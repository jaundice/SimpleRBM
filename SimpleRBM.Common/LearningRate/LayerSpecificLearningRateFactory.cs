using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleRBM.Common.LearningRate
{
    public class LayerSpecificLearningRateCalculatorFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly List<ILearningRateCalculatorFactory<T>> _innerFactories;

        public LayerSpecificLearningRateCalculatorFactory(IEnumerable<ILearningRateCalculatorFactory<T>> innerFactories)
        {
            _innerFactories = innerFactories.ToList();
        }

        public LayerSpecificLearningRateCalculatorFactory(params ILearningRateCalculatorFactory<T>[] innerFactories)
            : this(innerFactories.ToList())
        {
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return _innerFactories[layer].Create(layer);
        }
    }
}