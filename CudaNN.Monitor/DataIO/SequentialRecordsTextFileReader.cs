using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CudaNN.DeepBelief.DataIO
{
    public class SequentialRecordsTextFileReader<T> : TextFileReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public SequentialRecordsTextFileReader(LineReader<T> labelReader, LineReader<T> dataReader,
            bool firstLineIsHeaders,
            int totalRecordCount, string filePath, char fieldSeparator, int skipRecords)
            : base(
                labelReader, dataReader, firstLineIsHeaders, totalRecordCount, filePath, fieldSeparator)
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
                if (data.GetLength(0) > count)
                {
                    throw new Exception("More data than expected");
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
                if (data.GetLength(0) > count)
                {
                    throw new Exception("More data than expected");
                }
                return data;
            }
        }


        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded,
            out IList<string[]> labels)
        {
            var data =
               Enumerable.Range(0, (int)Math.Ceiling((double)count / batchSize))
                   .Select(
                       i =>
                           (i + 1) * batchSize < count
                               ? new T[batchSize, DataWidth]
                               : new T[count - ((i) * batchSize), DataWidth])
                   .ToList();


            var lbls = data.Select(a => new string[a.GetLength(0)]).ToList();
            var coded = data.Select(a => new T[a.GetLength(0), LabelDataWidth]).ToList();

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                for (var i = 0; i < data.Count; i++)
                {
                    for (var j = 0; j < data[i].GetLength(0); j++)
                    {

                        var line = sr.ReadLine();
                        string[] lineParts = line.Split(FieldSeparator);
                        LabelReader.CopyToTarget(coded[i], lineParts, j);
                        lbls[i][j] = string.Join(",", LabelReader.ReadTargetLineContents(coded[i], j));
                        DataReader.CopyToTarget(data[i], lineParts, j);

                    }
                }
                labelsEncoded = coded;
                labels = lbls;
                return data;
            }
        }

        public override IList<T[,]> Read(int count, int batchSize)
        {
            var data =
                Enumerable.Range(0, (int)Math.Ceiling((double)count / batchSize))
                    .Select(
                        i =>
                            (i + 1) * batchSize < count
                                ? new T[batchSize, DataWidth]
                                : new T[count - ((i) * batchSize), DataWidth])
                    .ToList();

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();
                for (int i = 0; i < SkipRecords; i++)
                    sr.ReadLine();

                for (var i = 0; i < data.Count; i++)
                {
                    for (var j = 0; j < data[i].GetLength(0); j++)
                    {

                        var line = sr.ReadLine();

                        string[] lineParts = line.Split(FieldSeparator);
                        DataReader.CopyToTarget(data[i], lineParts, j);
                    }
                }

                return data;
            }
        }

    }
}