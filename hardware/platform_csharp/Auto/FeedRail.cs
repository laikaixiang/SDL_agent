using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Winform_platform.Auto
{
    public class FeedRail
    {
        private static SerialPort rail_port = new SerialPort();

        /// <summary>
        /// returns true if port successfully opened
        /// </summary>
        /// <param name="port"></param>
        /// <param name="baudrate"></param>
        /// <returns></returns>
        public static bool connect(string port, int baudrate)
        {
            rail_port.PortName = port;
            rail_port.BaudRate = baudrate;
            try
            {
                rail_port.Open();
                return rail_port.IsOpen;
            }
            catch (Exception ex)
            {
                return false;
            }
        }

        /// <summary>
        /// returns true if port successfully closed
        /// </summary>
        /// <returns></returns>
        public static bool disconnect()
        {
            try
            {
                rail_port.Close();
                return !rail_port.IsOpen;
            }
            catch (Exception ex)
            {
                return false;
            }
        }

        /// <summary>
        /// changes the direction the feeding rail is moving
        /// </summary>
        public static void change_direction()
        {
            rail_port.Write("d");
            //return rail_port.ReadLine();
        }

        /// <summary>
        /// stops or starts the moving of the feeding rail
        /// </summary>
        public static void change_state()
        {
            rail_port.Write("r");
            //return rail_port.ReadLine();
        }
    }
}
