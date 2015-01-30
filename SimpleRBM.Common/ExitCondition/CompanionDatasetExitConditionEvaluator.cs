using System;
using System.Collections;
using System.Collections.Generic;

namespace SimpleRBM.Common.ExitCondition
{
    public class CompanionDatasetExitConditionEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private const int MAXGREATER = 7;
        private const int BUFFERSIZE = 500;
        private readonly CompanionDatasetExitConditionEvaluatorFactory<T> _factory;
        private readonly int _layer;
        private readonly int _maxEpochs;
        private readonly int _updateEpochs = 200;
        private T _biggestCompanionDelta;
        private T _biggestMainDelta;

        private CircularBuffer<T> _companionWindow;
        private CircularBuffer<T> _mainWindow;
        private int _numGreater;
        private bool _exitNextLowest;


        public CompanionDatasetExitConditionEvaluator(CompanionDatasetExitConditionEvaluatorFactory<T> factory,
            int layer, int updateEpochs, int maxEpochs)
        {
            _factory = factory;
            _layer = layer;
            _updateEpochs = updateEpochs;
            _maxEpochs = maxEpochs;

        }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime)
        {
            if (epochNumber > _maxEpochs)
                return true;



            var mainDelta = (T)Convert.ChangeType(
                ((double)Convert.ChangeType(lastError, typeof(double)) -
                 ((double)Convert.ChangeType(_mainWindow.MinValueSeen, typeof(double)))), typeof(T));

            _factory.MainTracker.LogEpochError(_layer, epochNumber, lastError);

            
            if (epochNumber % 20 == 0)
                if (epochNumber % 20 == 0)
                    Console.WriteLine("Epoch: {0}\tLayer: {1}\tError: {2}\tElapsed: {3}", epochNumber, _layer, lastError, elapsedTime);


      

            bool mainDownward = false;

            if (Comparer<T>.Default.Compare(lastError, _mainWindow.MinValueSeen) < 0)
            {
                mainDownward = true;
                _biggestMainDelta = default(T);
            }
            else
            {
                if (Comparer<T>.Default.Compare(lastError, _biggestCompanionDelta) > 0)
                {
                    _biggestMainDelta = mainDelta;
                }
            }

            _mainWindow.Add(lastError);


            if (epochNumber == 0 || mainDownward)
            {
                T companionError = _factory.Dbn.GetReconstructionError(_factory.TestData, _layer);

                var companionDelta = (T)Convert.ChangeType(
                    ((double)Convert.ChangeType(companionError, typeof(double)) -
                     ((double)Convert.ChangeType(_companionWindow.MinValueSeen, typeof(double)))), typeof(T));

                _factory.CompanionTracker.LogEpochError(_layer, epochNumber, companionError);

                Console.WriteLine("Epoch: {0} Companion error: {1}\t delta: {2}", epochNumber, companionError,
                    companionDelta);

                if (epochNumber > 0 && mainDownward)
                {

                    if (Comparer<T>.Default.Compare(companionError, _companionWindow.MinValueSeen) > 0)
                    {
                        _numGreater++;
                        bool companionDownward = true;
                        if (Comparer<T>.Default.Compare(companionDelta, _biggestCompanionDelta) > 0)
                        {
                            _biggestCompanionDelta = companionDelta;
                            companionDownward = false;
                            if (_exitNextLowest)
                                return true;
                        }

                        if (_numGreater > MAXGREATER)
                            _exitNextLowest = true;
                    }
                    else
                    {
                        if (_exitNextLowest)
                            return true;

                        _biggestCompanionDelta = default(T);
                        _numGreater = 0;
                    }
                    ////only exit if the main set is converging and the companion set is diverging
                    //if (mainDownward && shouldExit)
                    //    return true;


                    _companionWindow.Add(companionError);
                }
            }

            //if (epochNumber%_updateEpochs == 0)
            //{
            //    T companionError = _factory.Dbn.GetReconstructionError(_factory.TestData, _layer);

            //    var companionDelta = (T) Convert.ChangeType(
            //        ((double) Convert.ChangeType(companionError, typeof (double)) -
            //         ((double) Convert.ChangeType(_companionWindow.MinValueSeen, typeof (double)))), typeof (T));

            //    Console.WriteLine("Epoch: {0} Companion error: {1}\t delta: {2}", epochNumber, companionError,
            //        companionDelta);


            //    bool shouldExit = false;
            //    if (Comparer<T>.Default.Compare(companionError, _companionWindow.MinValueSeen) > 0)
            //    {
            //        _numGreater++;
            //        bool companionDownward = true;
            //        if (Comparer<T>.Default.Compare(companionDelta, _biggestCompanionDelta) > 0)
            //        {
            //            _biggestCompanionDelta = companionDelta;
            //            companionDownward = false;
            //            if (_exitNextLowest)
            //                return true;
            //        }

            //        if (_numGreater > MAXGREATER)
            //            _exitNextLowest = true;
            //    }
            //    else
            //    {
            //        if (_exitNextLowest)
            //            return true;

            //        _biggestCompanionDelta = default(T);
            //        _numGreater = 0;
            //    }
            //    ////only exit if the main set is converging and the companion set is diverging
            //    //if (mainDownward && shouldExit)
            //    //    return true;


            //    _companionWindow.Add(companionError);
            //}
            return false;
        }

        public void Reset()
        {
            _mainWindow = new CircularBuffer<T>(BUFFERSIZE);
            _companionWindow = new CircularBuffer<T>(BUFFERSIZE);
        }
    }

    public class CircularBuffer<T> : IEnumerable<T>
    {
        private readonly int _maxSize;
        private int _currSize;

        private Node _first;
        private Node _last;

        public CircularBuffer(int maxSize)
        {
            _maxSize = maxSize;
        }

        public T MinValueSeen { get; private set; }
        public T MaxValueSeen { get; private set; }

        public IEnumerator<T> GetEnumerator()
        {
            for (Node n = _first; n != null; n = n.NextNode)
            {
                yield return n.Value;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Add(T entry)
        {
            var node = new Node
            {
                Value = entry
            };

            if (_currSize == 0)
            {
                MinValueSeen = entry;
                MaxValueSeen = entry;

                _first = node;
                _last = node;
                _currSize++;
            }
            else
            {
                if (Comparer<T>.Default.Compare(MinValueSeen, entry) > 0)
                {
                    MinValueSeen = entry;
                }
                if (Comparer<T>.Default.Compare(entry, MaxValueSeen) > 0)
                {
                    MaxValueSeen = entry;
                }

                _last.NextNode = node;
                _last = node;
                _currSize++;
                if (_currSize > _maxSize)
                {
                    Node oldFirst = _first;
                    _first = oldFirst.NextNode;
                    _currSize--;
                    oldFirst.NextNode = null;
                }
            }
        }

        private class Node
        {
            public T Value { get; set; }
            public Node NextNode { get; set; }
        }
    }
}