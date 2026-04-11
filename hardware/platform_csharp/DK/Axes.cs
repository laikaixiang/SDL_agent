using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace WinFormsApp_Draft.DK
{
    class Axes
    {
        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int OpenComPort(byte index, uint baudRate);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int CloseComPort(byte index);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Read_Motor_position_c(byte index, byte id, ref int position_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Zero_c(byte index, byte id);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Zero_p(byte index, byte id);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_zero_c(byte index, byte id, ref byte zero_status);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_status_c(byte index, byte id, ref byte zero_status);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Find_status_p(byte index, byte id, ref byte zero_status);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Enable_motor_c(byte index, byte id, byte select_value);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Set_speed_ac_de_time_c(byte index, byte id, int speed, byte add_time, byte cut_time);

        [DllImport("DK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Motor_Absolute_movement_c(byte index, byte id, int Moveposition);
    }
}
