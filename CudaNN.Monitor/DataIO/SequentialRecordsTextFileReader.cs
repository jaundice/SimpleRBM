using System;
using System.Collections.Generic;
using System.IO;

namespace CudaNN.DeepBelief.DataIO
{
    public class SequentialRecordsTextFileReader<T> : TextFileReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public SequentialRecordsTextFileReader(LineReader<T> labelReader, LineReader<T> dataReader,
            bool firstLineIsHeaders,
            int totalRecordCount, string filePath, char fieldSeparator, int skipRecords)
            : base(labelReader, dataReader, firstLineIsHeaders, totalRecordCount, filePath, fieldSeparator)
        {
            SkipRecords = skipRecords;
        }

        public int SkipRecords { get; protected set; }

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels)
        {
            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                var data = new T[count, DataWidth];
                labelsEncoded = new T[count, LabelDataWidth];
                labels = new string[count];

                int lineNo = 0;
                string line;
                for (line = sr.ReadLine(), lineNo = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine(), lineNo++)
                {
                    string[] lineParts = line.Split(FieldSeparator);
                    LabelReader.CopyToTarget(labelsEncoded, lineParts, lineNo);
                    labels[lineNo] = string.Join(",", LabelReader.ReadTargetLineContents(labelsEncoded, lineNo));
                    DataReader.CopyToTarget(data, lineParts, lineNo);
                }

                return data;
            }
        }

        public override T[,] Read(int count)
        {
            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                var data = new T[count, DataWidth];

                int lineNo = 0;
                string line;
                for (line = sr.ReadLine(), lineNo = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine(), lineNo++)
                {
                    string[] lineParts = line.Split(FieldSeparator);
                    DataReader.CopyToTarget(data, lineParts, lineNo);
                }

                return data;
            }
        }


        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded,
            out IList<string[]> labels)
        {

            List<string[]> lbls = new List<string[]>();
            List<T[,]> coded = new List<T[,]>();
            List<T[,]> data = new List<T[,]>();

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                var dat = new T[batchSize, DataWidth];
                var cod = new T[batchSize, LabelDataWidth];
                var lb = new string[batchSize];

                data.Add(dat);
                coded.Add(cod);
                lbls.Add(lb);

                int lineNo = 0;
                string line;
                int j = 0;
                for (line = sr.ReadLine(), lineNo = 0, j = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine(), lineNo++, j++)
                {
                    string[] lineParts = line.Split(FieldSeparator);
                    LabelReader.CopyToTarget(cod, lineParts, j);
                    lb[j] = string.Join(",", LabelReader.ReadTargetLineContents(cod, j));
                    DataReader.CopyToTarget(dat, lineParts, j);

                    if (j == dat.GetLength(0) - 1)
                    {
                        j = 0;
                        var newbatchSize = lineNo + batchSize < count ? batchSize : count - lineNo;
                        if (newbatchSize > 0)
                        {
                            dat = new T[newbatchSize, DataWidth];
                            lb = new string[newbatchSize];
                            cod = new T[newbatchSize, LabelDataWidth];
                            data.Add(dat);
                            lbls.Add(lb);
                            coded.Add(cod);
                        }
                    }
                }

                labelsEncoded = coded;
                labels = lbls;
                return data;
            }
        }

        public override IList<T[,]> Read(int count, int batchSize)
        {
            List<T[,]> data = new List<T[,]>();

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                var dat = new T[batchSize, DataWidth];

                data.Add(dat);

                int lineNo = 0;
                string line;
                int j = 0;
                for (line = sr.ReadLine(), lineNo = 0, j = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine(), lineNo++, j++)
                {
                    string[] lineParts = line.Split(FieldSeparator);
                    DataReader.CopyToTarget(dat, lineParts, j);

                    if (j == dat.GetLength(0) - 1)
                    {
                        j = 0;
                        var newbatchSize = lineNo + batchSize < count ? batchSize : count - lineNo;
                        if (newbatchSize > 0)
                        {
                            dat = new T[newbatchSize, DataWidth];
                            data.Add(dat);
                        }
                    }
                }
                return data;
            }
        }

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels, Func<T, T> sourceToTargetConverter)
        {
            throw new NotImplementedException();
        }

        public override T[,] Read(int count, Func<T, T> sourceToTargetConverter)
        {
            throw new NotImplementedException();
        }

        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels, Func<T, T> sourceToTargetConverter)
        {
            throw new NotImplementedException();
        }

        public override IList<T[,]> Read(int count, int batchSize, Func<T, T> sourceToTargetConverter)
        {
            throw new NotImplementedException();
        }
    }
}