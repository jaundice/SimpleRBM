namespace SimpleRBM.Common
{
    public class LayerDefinition : ILayerDefinition
    {
        public int VisibleUnits { get; set; }

        public int HiddenUnits { get; set; }

        public ActivationFunction VisibleActivation { get; set; }

        public ActivationFunction HiddenActivation { get; set; }
    }
}