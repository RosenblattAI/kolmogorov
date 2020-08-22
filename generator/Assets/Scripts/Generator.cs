using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Threading; // for hacky solution

// Author: Ryan James Walden

[RequireComponent(typeof(Camera))]
public class Generator : MonoBehaviour
{
    // Can probably remove these for OOP sake
    [SerializeField] private Transform _sunTransform;
    private Texture2D _screenshotTexture;
    private RenderTexture _renderTexture;
    private static string outputPath;
    private Camera _camera;

    /* Fun things to consider:
        - Variation control
        - Caching
        - S3 / Cloud Storage
        - Lossless compression (crop to obj), save removed borders shape
        - Progress bar
        - More error handling and logging
        - Lighting variations
        - Artifiacts / celestial objects
        - Headless mode
    */


    void Start()
    {
        // Grab arguments
        string[] args = System.Environment.GetCommandLineArgs();

        // Setup default arguments
        int resWidth = 224;
        int resHeight = 224;
        int samples = 1000;

        // Get args type
        var argsType = args.GetType();

        // Verify args type and args' element type
        if (argsType.IsArray)
        {
            // Verify all arguments are present
            if (args.Length != 3)
            {
                Debug.Log("Missing arguments");
            }
            else
            {
                // Setup arguments
                resWidth = System.Int32.Parse(args[0]);
                resHeight = System.Int32.Parse(args[1]);
                samples = System.Int32.Parse(args[2]);

                Debug.Log("Using custom arguments");
            }
        }
        else
        {
            Debug.Log("No arguments found");
        }

        _camera = GetComponent<Camera>();

         _sunTransform = GameObject.Find("Sun").transform;

        // Setup the observatory
        SetupTextures(resWidth, resHeight);

        StartCoroutine(GenerateSatellitesAndTakeImages(resWidth, resHeight, samples));
    }

    private IEnumerator GenerateSatellitesAndTakeImages(int resWidth, int resHeight, int samples)
    {

        // Get all GameObjects
        Object[] satelliteObjects = Resources.LoadAll("Satellites", typeof(GameObject));

        // Iterate through each object
        foreach (GameObject satellite in satelliteObjects)
        {
            // Update the console log
            Debug.Log("Rendering satellite: " + satellite.name);

            // Add a satellite to the scene
            GameObject mySatellite = Instantiate(satellite, Vector3.zero, Quaternion.identity);

            // Create directory for satellite if none exists
            outputPath = "Dataset/" + satellite.name;
            if (!Directory.Exists(outputPath)) Directory.CreateDirectory(outputPath);

            // Call the generation function
            //StartCoroutine(GenerateImages(mySatellite, resWidth, resHeight, samples));
            yield return StartCoroutine(GenerateImages(mySatellite, resWidth, resHeight, samples));

            // Remove the current Satellite from the scene, could make a corountine
            mySatellite.SetActive(false);
            Destroy(mySatellite);
            yield return null;
        }

        // Clean up camera
        CleanupObservatory();

        // Exit
        Debug.Log("Synthetic data generation complete");
        Application.Quit();
    }

    private void SetupTextures(int width, int height)
    {
        _renderTexture = new RenderTexture(width, height, 24);
        _screenshotTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        Debug.Log("Observatory setup complete");
    }

    Bounds CalculateBounds(GameObject obj)
    {
        Bounds b = new Bounds(obj.transform.position, Vector3.zero);
        Object[] rList = obj.GetComponentsInChildren(typeof(Renderer));
        foreach (Renderer r in rList)
        {
            b.Encapsulate(r.bounds);
        }
        return b;
    }

    void FocusObservatoryOnSatellite(GameObject satellite, float magnification = 1.0f)
    {
        Bounds b = CalculateBounds(satellite);
        Vector3 max = b.size;
        float radius = Mathf.Max(max.x, Mathf.Max(max.y, max.z));
        float dist = radius / (Mathf.Sin(_camera.fieldOfView * Mathf.Deg2Rad / 2f));
        _camera.transform.position = new Vector3(0, 0, 1) * dist / magnification + b.center;
        _camera.transform.LookAt(b.center);
    }


    private void CleanupObservatory()
    {
        _camera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(_renderTexture);
        Debug.Log("Observatory cleanup complete");
    }

    private Vector3 GetSatelliteOrientation(GameObject Satellite)
    {
        // get relative orientation of satellite from observatory
        // Get rotation necessary to orient satellite towards observatory
        // (if satellite was rotated with the calculated vector, the appearance would depend on neutral orientation of satellite)
        // incorporate the "up" direction of the observatory

        var direction = _camera.transform.position - Satellite.transform.position;
        Quaternion quat_rotation = Quaternion.LookRotation(direction, Satellite.transform.up);
        Vector3 rotation = quat_rotation.eulerAngles;
        // Debug.Log("Relative Obs-Sat angle: " + rotation);

        return rotation;
    }

    private string EulerAngleToString(Vector3 orientation)
    {
        return string.Format("{0},{1},{2}", orientation.x.ToString(), orientation.y.ToString(), orientation.z.ToString());
    }


    //IEnumerator
    private IEnumerator GenerateImages(GameObject Satellite, int width, int height, int iterations)
    {
        int angleStep = 30; // degrees
        int iteration = 0;
        Quaternion initRotation = Satellite.transform.rotation;
        for (int i = 0; i < 360/angleStep; i++)
        {
            for (int j = 0; j < 360/angleStep; j++)
            {
                for (int k = 0; k < 360/angleStep; k++)
                {
                    // Randomly orient the satellite
                    Debug.Log(string.Format("{0},{1},{2}", i * angleStep, j * angleStep, k * angleStep));
                    Satellite.transform.rotation = initRotation;
                    Vector3 toRotate = new Vector3(i * angleStep,
                                                        j * angleStep,
                                                        k * angleStep);
                    Satellite.transform.Rotate(toRotate);

                    // Focus the camera on the object
                    FocusObservatoryOnSatellite(Satellite, 1f);
                    string orientation = EulerAngleToString(toRotate);

                    // Focus the sun on the object
                    _sunTransform.rotation = _camera.transform.rotation;

                    float distance = Vector3.Distance(_camera.transform.position, Satellite.transform.position);

                    // Probably should change this
                    // get main camera and manually render scene into rt
                    _camera.targetTexture = _renderTexture;
                    _camera.Render();

                    // read pixels will read from the currently active render texture so make our offscreen 
                    // render texture active and then read the pixels
                    RenderTexture.active = _renderTexture;

                    // Take a screenshot in 224x224 resolution
                    // AsyncGPUReadback may be preffered to ReadPixels
                    _screenshotTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);

                    // Save the screenshot to PNG in the corresponding satellite directory
                    byte[] bytes = _screenshotTexture.EncodeToPNG();
                    string filename = string.Format(
                        outputPath + "/{0}_{1}x{2}_{3}_{4}.png",
                        iteration,
                        width,
                        height,
                        orientation,
                        distance);
                    File.WriteAllBytes(filename, bytes);
                    iteration++;
                    yield return null;
                }
            }
        }
    }
}
