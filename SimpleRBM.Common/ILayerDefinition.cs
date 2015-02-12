namespace SimpleRBM.Common
{
    public interface ILayerDefinition
    {
        int VisibleUnits { get; }
        int HiddenUnits { get; }
        ActivationFunction VisibleActivation { get; }
        ActivationFunction HiddenActivation { get; }

    }

    public enum ActivationFunction
    {
        SoftPlus,
        Sigmoid,
        Tanh,
        SoftMax
    }
}