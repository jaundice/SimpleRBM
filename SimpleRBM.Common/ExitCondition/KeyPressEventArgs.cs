using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimpleRBM.Common.ExitCondition
{
    public class KeyPressEventArgs : EventArgs
    {
        public KeyPressEventArgs(ConsoleKeyInfo info)
        {
            KeyInfo = info;
        }

        public ConsoleKeyInfo KeyInfo { get; protected set; }
    }
}
