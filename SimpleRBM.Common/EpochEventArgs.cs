using System;

namespace SimpleRBM.Common
{
    public class EpochEventArgs<T> : EventArgs
    {
        public int Layer { get; set; }
        public int Epoch { get; set; }
        public T Error { get; set; }
        public T Delta { get; set; }
        public T LearningRate { get; set; }
        public TimeSpan Elapsed { get; set; }
    }
}