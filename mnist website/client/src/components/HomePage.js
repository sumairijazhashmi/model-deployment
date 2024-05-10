
import { useState } from 'react';
import axios from 'axios';


function HomePage() {

    const [file, setFile] = useState(null);
    const [result, setResult] = useState(-1);

    // function to upload file
    const fileUpload = (event) => {
        setFile(event.target.files[0]);
    }

    // function to send image to backend
    const upload = async () => {
        if (!file) {
            return;
        }

        // error handling file format
        if (file) {
            const fileExtension =file.name.split('.').pop().toLowerCase();
            if (!['png', 'jpeg', 'jpg'].includes(fileExtension)) {
                alert('Only PNG, JPEG, and JPG files are allowed.');
                return;
            }
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setResult(response.data.predicted_class);
        }
        catch (err) {
            console.log("error");
        }
    }
    
    return (
        <div>
            <h1>MNIST Digit Classifier</h1>
            <p>
                Hi, I made this website to learn how to deploy ML models. 
                <br></br> 
                Please upload an image of a digit in this form to obtain the numeric digit.
            </p>
            <input type="file" onChange={fileUpload} />
            <button onClick={upload}>Upload</button>

            <h2>Result:</h2>
            <p>{result}</p>
        </div>
    )
}

export default HomePage;