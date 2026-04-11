using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CSharpTcpDemo.com.dobot.api;
using DocumentFormat.OpenXml.Spreadsheet;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using uPLibrary.Networking.M2Mqtt;

namespace Winform_platform.Auto
{
    class Agent
    {
        public const string topic = "do_experiment";
        public static List<string> step_buffer = new List<string>();

        public static void to_step_buffer(string parameters)
        {
            step_buffer.Add(parameters);
        }
        public static void clear_step_buffer()
        {
            step_buffer.Clear();
        }
    }

    class Platform_Config
    {

        public static void record_reagent(string json_path, string pos, string reagent)
        {
            try
            {
                string reagent_json = File.ReadAllText(json_path);
                JObject json = JObject.Parse(reagent_json);
                json["Points"][pos]["name"] = reagent;
                File.WriteAllText(json_path, json.ToString(Formatting.Indented));
            }
            catch { }
        }

        public static void clear_all_reagents(string json_path)
        {
            try
            {   
                string reagent_json = File.ReadAllText(json_path);
                JObject json = JObject.Parse(reagent_json);
                for(int i = 1; i < 7; i++)
                {
                    for(int j = 1; j < 5; j++)
                    {
                        json["Points"][$"BP{i}{j}"]["name"] = "";
                        File.WriteAllText(json_path, json.ToString(Formatting.Indented));
                    }
                }
            }
            catch { }
        }
    }
}
