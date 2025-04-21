
document.addEventListener('DOMContentLoaded', function() {
    // Connect to WebSocket
    const socket = new WebSocket(`ws://${window.location.host}/ws`);

    socket.onopen = function(e) {
        console.log('WebSocket connection established');
    };

    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.event === 'all_recommendations') {
                console.log('Received recommendations data:', data);
                // You could implement a network visualization here
                // For now, we'll just show a message
                document.getElementById('network-container').innerHTML = 
                    '<p>Data received: ' + Object.keys(data.data).length + 
                    ' patient recommendations loaded.</p>';
            }
        } catch (e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };

    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };

    socket.onclose = function(event) {
        console.log('WebSocket connection closed');
    };
});
        