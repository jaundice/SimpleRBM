namespace SimpleRBM.Common
{
    public interface ILayerSaveInfo<T>
    {
        int NumVisible { get; }
        int NumHidden { get; }
        T[,] Weights { get; }
        ActivationFunction VisibleActivation { get; }
        ActivationFunction HiddenActivation { get; }
        void Save(string filePath);
    }
}