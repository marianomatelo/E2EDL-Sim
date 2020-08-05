import sys
import glob
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *


def get_image():
    '''
    Reads a RGB image from AirSim and prepare it for consumption by the model
    '''
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

    return image_rgba[76:135, 0:255, 0:3].astype(float)


if __name__ == '__main__':

    print('=====================================================')
    print('                 STARTING TESTING                    ')
    print('=====================================================')

    # Model with the lowest validation loss from training will be used
    models = glob.glob('model/models/*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

    # Loads trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Connect to AirSim, AirSim must be running on background
    client = CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = CarControls()

    # Starting parameters
    car_controls.steering = 0
    car_controls.throttle = 0
    car_controls.brake = 0

    image_buf = np.zeros((1, 59, 255, 3))
    state_buf = np.zeros((1, 4))

    while (True):
        car_state = client.getCarState()

        # if (car_state.speed < 5):
        #     car_controls.throttle = 1.0
        # else:
        #     car_controls.throttle = 0.0

        image_buf[0] = get_image()
        state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
        model_output = model.predict([image_buf, state_buf])
        car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

        print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))

        client.setCarControls(car_controls)

    print('=====================================================')
    print('                TESTING COMPLETED                    ')
    print('=====================================================')