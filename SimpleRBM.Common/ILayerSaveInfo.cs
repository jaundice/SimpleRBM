namespace SimpleRBM.Common
{
    public interface ILayerSaveInfo<T>
    {
        int NumVisible { get; }
        int NumHidden { get; }
        T[,] Weights { get; }
        void Save(string filePath);
    }
}