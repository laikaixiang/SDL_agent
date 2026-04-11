using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using uPLibrary.Networking.M2Mqtt;
using uPLibrary.Networking.M2Mqtt.Messages;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.StartPanel;

namespace WinFormsApp_Draft.Auto
{
    public class Mqtt_connection
    {
        public static string msg = "none";

        public static MqttClient ConnectMQTT(MqttClient client, string clientId, string username, string password)
        {
            for (int i = 0; i < 4; i++)
            {
                if (Connection_check(client))
                {
                    break;
                }
                client.Connect(clientId, username, password);
            }
            return client;
        }

        public static MqttClient DisconnectMQTT(MqttClient client)
        {
            for (int i = 0; i < 4; i++)
            {
                if (!Connection_check(client))
                {
                    break;
                }
                client.Disconnect();
            }

            return client;
        }

        public static bool Connection_check(MqttClient client)
        {
            return client.IsConnected;
        }

        public static void Publish(MqttClient client, string topic, string msg)
        {
            client.Publish(topic, System.Text.Encoding.UTF8.GetBytes(msg), MqttMsgBase.QOS_LEVEL_EXACTLY_ONCE, false);
        }

        public static void Subscribe(MqttClient client, string topic)
        {
            client.MqttMsgPublishReceived += MqttMsgReceived;
            client.Subscribe(new string[] { topic }, new byte[] { MqttMsgBase.QOS_LEVEL_EXACTLY_ONCE });
        }

        public static void MqttMsgReceived(object sender, MqttMsgPublishEventArgs e)
        {
            string payload = System.Text.Encoding.Default.GetString(e.Message);
            msg = payload.ToString();
        }

        public static void clear_msg()
        {
            msg = "none";
        }

        public static void Unsubscribe(MqttClient client, string topic)
        {
            client.MqttMsgPublishReceived -= MqttMsgReceived;
            client.Unsubscribe(new string[] { topic });
        }

        /// <summary>
        /// wait for data analyzing client to respond. 
        /// if it is ready for the next message, will receive 'next' and quit while loop
        /// </summary>
        /// <returns></returns>
        public static async Task wait()
        {
            while (msg != "next")
            {
                await Task.Delay(100);
            }
            msg = "none";
        }
    }
}
