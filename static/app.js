document.getElementById('prediction-form').addEventListener('submit', function(event) {
  event.preventDefault();
  
  let weight = document.getElementById('weight').value;
  
  fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `weight=${weight}`
  })
  .then(response => response.json())
  .then(data => {
      document.getElementById('result').innerHTML = `Predicted Height: ${data.predicted_height.toFixed(2)} cm`;
  })
  .catch(error => console.error('Error:', error));
});
