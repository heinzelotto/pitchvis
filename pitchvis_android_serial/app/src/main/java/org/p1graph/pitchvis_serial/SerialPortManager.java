package org.p1graph.pitchvis_serial;

import android.content.Context;
import android.hardware.usb.UsbManager;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.util.Log;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import java.io.IOException;
import java.util.List;

public class SerialPortManager {
    private UsbSerialPort port;
    private UsbManager usbManager;

    public SerialPortManager(UsbManager usbManager) {
        this.usbManager = usbManager;
    }

    public String openSerialPort() throws IOException {
        // Discovery
        List<UsbSerialDriver> availableDrivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);
        if (availableDrivers.isEmpty()) {
            // print in logcat
            Log.d("SerialPortManager", "No available drivers");

            throw new RuntimeException("No available drivers");
            //return "No available drivers";
        }

        // Open a connection to the first available driver.
        UsbSerialDriver driver = availableDrivers.get(0);
        UsbDeviceConnection connection = usbManager.openDevice(driver.getDevice());
        if (connection == null) {
            // permission handling: UsbManager.requestPermission(driver.getDevice(), ..) handling
            // var device = driver.getDevice();
            UsbDevice device = driver.getDevice();
            if (!usbManager.hasPermission(device)) {
                usbManager.requestPermission(device, null);
                boolean hasPermision = usbManager.hasPermission(device);

                if (!hasPermision) {
                    // print in logcat
                    Log.d("SerialPortManager", "No permission to open the serial port");

                    throw new RuntimeException("No permission to open the serial port");
                    //return "No permission to open the serial port";
                }
            }
        }

        port = driver.getPorts().get(0); // Most devices have just one port (port 0)
        try {
            port.open(connection);
            port.setDTR(true);
            port.setParameters(115200, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE);
        } catch (Exception e) {
            e.printStackTrace();
            // print in logcat
            Log.d("SerialPortManager", "Error opening the serial port: " + e.getMessage());
            throw new RuntimeException("Error opening the serial port: " + e.getMessage());
            //return "Error opening the serial port: " + e.getMessage();
        }
        
        return "Serial port opened successfully";
    }

    public void writeData(byte[] bytes) throws IOException {
        if (port != null) {
            try {
                port.write(bytes, 1000);
            } catch (Exception e) {
                e.printStackTrace();
                // print in logcat
                Log.d("SerialPortManager", "Error writing to the serial port: " + e.getMessage());
            }
        }
    }
}
