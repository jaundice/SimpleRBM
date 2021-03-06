﻿using System;
using System.IO;

namespace SimpleRBM.Common.ExitCondition
{
    public class EpochErrorFileTracker<T> : IEpochErrorTracker<T>, IDisposable
    {
        private readonly TextWriter _writer;

        public EpochErrorFileTracker(string filePath)
        {
            _writer = new StreamWriter(File.Open(filePath, FileMode.Create, FileAccess.Write, FileShare.Read));
            _writer.WriteLine("layer\tepoch\terror\tdelta\telapsed");
            _writer.Flush();
        }

        public bool Disposed { get; private set; }

        public void Dispose()
        {
            if (!Disposed)
            {
                Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        public void LogEpochError(int layer, int epoch, T error, T delta, TimeSpan elapsed)
        {
            _writer.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}", layer, epoch, error, delta, elapsed);
            _writer.Flush();
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                _writer.Close();
            }
        }
    }
}