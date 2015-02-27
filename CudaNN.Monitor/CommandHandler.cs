using System;
using System.Windows.Input;

namespace CudaNN.Monitor
{
    public class CommandHandler : ICommand
    {
        private readonly Action<object> _action;
        private readonly Func<object, bool> _canRun;

        public CommandHandler(Action<object> action, Func<object, bool> canRun)
        {
            _action = action;
            _canRun = canRun;
        }

        public bool CanExecute(object parameter)
        {
            return _canRun(parameter);
        }

        public event EventHandler CanExecuteChanged;

        public void Execute(object parameter)
        {
            _action(parameter);
        }
    }
}