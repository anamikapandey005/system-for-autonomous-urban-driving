const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');

// DOM Elements
const valSpeed = document.getElementById('val-speed');
const barSpeed = document.getElementById('bar-speed');
const valAccel = document.getElementById('val-accel');
const barAccel = document.getElementById('bar-accel');
const valSteer = document.getElementById('val-steer');

// Coordinate Mapping
const scale = 8; // Pixels per meter
const offsetX = 100;
const offsetY = 200;

function worldToScreen(x, y) {
    // Center ego around 200px from left
    return {
        x: x * scale + offsetX,
        y: y * scale + offsetY
    };
}

function drawGrid() {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for(let i=0; i<800; i+=40) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 400); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(800, i); ctx.stroke();
    }
}

function drawRoad() {
    // Main horizontal road
    ctx.fillStyle = '#334155';
    ctx.fillRect(0, offsetY - 40, 800, 80);
    
    // Intersection cross road
    ctx.fillRect(worldToScreen(50, 0).x - 40, 0, 80, 400);
    
    // Dashed center lines
    ctx.strokeStyle = '#fde047';
    ctx.setLineDash([15, 15]);
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, offsetY); ctx.lineTo(worldToScreen(50,0).x - 40, offsetY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(worldToScreen(50,0).x + 40, offsetY); ctx.lineTo(800, offsetY); ctx.stroke();
    ctx.setLineDash([]);
}

function renderState(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Dynamic camera tracking ego x position
    const cameraX = state.ego.x;
    
    // Offset calculation for drawing
    function w2s(x, y) {
        return {
            x: (x - cameraX) * scale + 200,
            y: y * scale + offsetY
        };
    }

    drawRoad(); // Simplified road, in a real implementation we would shift road with camera
    
    // Main road (shifting)
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0,0,800,400);
    ctx.fillStyle = '#334155';
    ctx.fillRect(0, offsetY - 60, 800, 120); // Road
    
    // Draw crossroad exactly where the traffic light is
    state.traffic_lights.forEach(tl => {
        ctx.fillStyle = '#334155';
        ctx.fillRect(w2s(tl.x, 0).x - 60, 0, 120, 400); // Crossroad
    });

    // Traffic Lights
    state.traffic_lights.forEach(tl => {
        const pos = w2s(tl.x, tl.y);
        ctx.fillStyle = tl.state === 'RED' ? '#ef4444' : tl.state === 'YELLOW' ? '#eab308' : '#10b981';
        ctx.beginPath();
        ctx.arc(pos.x, offsetY - 70, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 15;
        ctx.shadowColor = ctx.fillStyle;
        ctx.fill();
        ctx.shadowBlur = 0; // Reset
        
        // Stop line
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(pos.x, offsetY - 60); ctx.lineTo(pos.x, offsetY + 60); ctx.stroke();
    });

    // Stop Signs
    state.stop_signs.forEach(sign => {
        const pos = w2s(sign.x, sign.y);
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(pos.x, offsetY - 70, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = '8px Arial';
        ctx.fillText('STOP', pos.x - 8, offsetY - 67);
        // Stop line
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(pos.x, offsetY - 60); ctx.lineTo(pos.x, offsetY + 60); ctx.stroke();
    });

    // Planned Trajectory
    if (state.trajectory.length > 0) {
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(w2s(state.trajectory[0][0], state.trajectory[0][1]).x, w2s(state.trajectory[0][0], state.trajectory[0][1]).y);
        state.trajectory.forEach(p => {
            const pos = w2s(p[0], p[1]);
            ctx.lineTo(pos.x, pos.y);
        });
        ctx.stroke();
    }

    // Ego Vehicle
    const egoPos = w2s(state.ego.x, state.ego.y);
    ctx.save();
    ctx.translate(egoPos.x, egoPos.y);
    ctx.rotate(state.ego.yaw);
    ctx.fillStyle = '#3b82f6'; // Glowing blue
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#3b82f6';
    ctx.fillRect(-20, -10, 40, 20); // Draw Car
    ctx.shadowBlur = 0;
    ctx.restore();

    // Agents
    state.agents.forEach(agent => {
        const pos = w2s(agent.x, agent.y);
        if(agent.type === 'pedestrian') {
            ctx.fillStyle = '#ef4444';
            ctx.beginPath(); ctx.arc(pos.x, pos.y, 6, 0, Math.PI*2); ctx.fill();
        } else {
            ctx.fillStyle = '#f59e0b';
            ctx.fillRect(pos.x - 15, pos.y - 10, 30, 20);
        }
    });

    // Predictions
    if(state.predictions) {
        Object.values(state.predictions).forEach(pred => {
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 2;
            ctx.beginPath();
            if(pred.preds.length > 0){
                ctx.moveTo(w2s(pred.preds[0][0], pred.preds[0][1]).x, w2s(pred.preds[0][0], pred.preds[0][1]).y);
                pred.preds.forEach(p => ctx.lineTo(w2s(p[0], p[1]).x, w2s(p[0], p[1]).y));
                ctx.stroke();
            }
            ctx.setLineDash([]);
        });
    }
}

function updateTelemetry(state) {
    // Speed
    valSpeed.innerHTML = `${state.ego.v.toFixed(1)} <span class="unit">m/s</span>`;
    barSpeed.style.width = `${Math.min((state.ego.v / 20) * 100, 100)}%`;
    
    // Acceleration
    valAccel.innerHTML = `${state.ego.a.toFixed(1)} <span class="unit">m/s²</span>`;
    // Map -4 to 4 accel to 0-100%
    const accelPercent = ((state.ego.a + 4) / 8) * 100;
    barAccel.style.width = `${Math.max(0, Math.min(accelPercent, 100))}%`;
    barAccel.style.background = state.ego.a < 0 ? '#ef4444' : '#10b981';

    // Steering
    valSteer.innerHTML = `${(state.ego.steer * (180/Math.PI)).toFixed(1)} <span class="unit">°</span>`;
}

// Fetch loop
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (response.ok) {
            const state = await response.json();
            if (state.ego) {
                renderState(state);
                updateTelemetry(state);
            }
        }
    } catch (e) {
        console.error("Connection lost");
    }
    requestAnimationFrame(fetchState);
}

// Start loop
fetchState();
