namespace SimpleRBM.Common
{
    public interface ILayerDefinition
    {
        int VisibleUnits { get; }
        int HiddenUnits { get; }
    }
}