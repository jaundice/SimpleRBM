using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows;
using SimpleRBM.Common;

namespace CudaNN.Monitor
{
    public class LayerSpecificLearningRateCalculatorFactory<T> : DependencyObject, ILearningRateCalculatorFactory<T>
    {
        public ObservableCollection<InteractiveLearningRateCalculatorFactory<T>> InnerCalculators { get; set; }

        public LayerSpecificLearningRateCalculatorFactory(IEnumerable<InteractiveLearningRateCalculatorFactory<T>> innerEvaluators)
        {
            InnerCalculators = new ObservableCollection<InteractiveLearningRateCalculatorFactory<T>>(innerEvaluators);
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return InnerCalculators[layer].Create(layer);
        }
    }
}