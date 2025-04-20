const COLOR_MAP = {
  W: "white",
  O: "orange",
  G: "green",
  R: "red",
  B: "blue",
  Y: "yellow",
};

function drawCube(cubeReduced) {
  const canvas = document.getElementById("cube-canvas");
  const ctx = canvas.getContext("2d");
  const tileSize = 30;
  const gap = 3;
  const borderWidth = 1;

const faces = [
  { start: [4, 1], blocks: [[0, 1], [2, 3]] },
  { start: [2, 3], blocks: [[4, 5], [12, 13]] },
  { start: [4, 3], blocks: [[6, 7], [14, 15]] },
  { start: [6, 3], blocks: [[8, 9], [16, 17]] },
  { start: [8, 3], blocks: [[10, 11], [18, 19]] },
  { start: [4, 5], blocks: [[20, 21], [22, 23]] },
];

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  faces.forEach((face) => {
      const startX = face.start[0] * (tileSize + gap);
      const startY = face.start[1] * (tileSize + gap);

      face.blocks.forEach((row, rowIndex) => {
          row.forEach((index, colIndex) => {
              const color = COLOR_MAP[cubeReduced[index]];
              const x = startX + colIndex * tileSize;
              const y = startY + rowIndex * tileSize;

              ctx.fillStyle = color;
              ctx.fillRect(x, y, tileSize, tileSize);
          });
      });

      ctx.strokeStyle = "#000000";
      ctx.lineWidth = borderWidth;

      face.blocks.forEach((row, rowIndex) => {
          row.forEach((index, colIndex) => {
              const x = startX + colIndex * tileSize;
              const y = startY + rowIndex * tileSize;

              if (colIndex > 0) {
                  ctx.beginPath();
                  ctx.moveTo(x, y);
                  ctx.lineTo(x, y + tileSize);
                  ctx.stroke();
              }
              if (rowIndex > 0) {
                  ctx.beginPath();
                  ctx.moveTo(x, y);
                  ctx.lineTo(x + tileSize, y);
                  ctx.stroke();
              }
          });
      });

      ctx.strokeRect(startX, startY, tileSize * 2, tileSize * 2);
  });
}

let isSolving = false;

function cancelSolution() {
  if (isSolving) {
      document.getElementById("solution").textContent = "当前阶段不可打断";
      return false;
  } else {
      if (solutionInterval !== null) {
          clearInterval(solutionInterval);
          clearTimeout(heatmapTimeout);
          solutionInterval = null;
          return true;
      }
      return true;
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const scrambleBtn = document.getElementById("scramble-btn");
  const solveBtn = document.getElementById("solve-btn");
  const resetBtn = document.getElementById("reset-btn");
  const solutionPre = document.getElementById("solution");
  const next_step_btn = document.getElementById("next-step-btn");
  const modeBtn = document.getElementById("next-step-mode");

  document.getElementById("f-btn").addEventListener("click", function () {
      simulateKeyPress("f");
  });
  document.getElementById("r-btn").addEventListener("click", function () {
      simulateKeyPress("r");
  });
  document.getElementById("u-btn").addEventListener("click", function () {
      simulateKeyPress("u");
  });
  document.getElementById("f-prime-btn").addEventListener("click", function () {
      simulateKeyPress("g");
  });
  document.getElementById("r-prime-btn").addEventListener("click", function () {
      simulateKeyPress("t");
  });
  document.getElementById("u-prime-btn").addEventListener("click", function () {
      simulateKeyPress("i");
  });

  drawCube(initialState);

  scrambleBtn.addEventListener("click", function () {
      if (!cancelSolution()) {
          return;
      }
      resetButtonColors();
      const steps = parseInt(document.getElementById("scramble-steps").value);

      fetch("/scramble", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: steps }),
      })
          .then((response) => response.json())
          .then((data) => {
              drawCube(data.cube_reduced);
              solutionPre.textContent = `打乱序列: ${data.scramble}`;
          });
  });

  solveBtn.addEventListener("click", function () {
      resetButtonColors();
      isSolving = true;
      const mode = modeBtn.value;

      if (mode === "full") {
          fetch("/check_solved", { method: "POST" })
              .then((response) => response.json())
              .then((data) => {
                  if (data.is_solved) {
                      solutionPre.textContent = "魔方已经还原";
                      isSolving = false;
                  } else {
                      fetch("/solve", { method: "POST" })
                          .then((response) => response.json())
                          .then((data) => {
                              if (data.solution.length > 0) {
                                  animateSolution(data.solution);
                                  heatmapTimeout = setTimeout(() => {
                                      if (data.solution_process && data.solution_process.length > 0) {
                                          drawHeatmap(data.solution_process);
                                      }
                                  }, data.solution.length * 100 + 200);
                                  isSolving = false;
                              } else {
                                  solutionPre.textContent = "未在50步内找到解法";
                                  drawCube(data.cube_reduced);
                                  isSolving = false;
                              }
                          });
                  }
              });
      } else {
          fetch("/check_solved_one_face", { method: "POST" })
              .then((response) => response.json())
              .then((data) => {
                  if (data.is_face_solved) {
                      solutionPre.textContent = "已有一面还原";
                      isSolving = false;
                  } else {
                      fetch("/solve_one_face", { method: "POST" })
                          .then((response) => response.json())
                          .then((data) => {
                              if (data.solution.length > 0) {
                                  animateSolution(data.solution);
                                  heatmapTimeout = setTimeout(() => {
                                      if (data.solution_process && data.solution_process.length > 0) {
                                          drawHeatmap(data.solution_process);
                                      }
                                  }, data.solution.length * 100 + 200);
                                  isSolving = false;
                              } else {
                                  solutionPre.textContent = "未在10步内找到解法";
                                  drawCube(data.cube_reduced);
                                  isSolving = false;
                              }
                          });
                  }
              });
      }
  });

  resetBtn.addEventListener("click", function () {
      if (!cancelSolution()) {
          return;
      }
      resetButtonColors();
      fetch("/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
      })
          .then((response) => response.json())
          .then((data) => {
              drawCube(data.cube_reduced);
              solutionPre.textContent = "魔方已复原";
          });
  });

  next_step_btn.addEventListener("click", function () {
      if (!cancelSolution()) {
          return;
      }
      const mode = modeBtn.value;

      if (mode === "full") {
          fetch("/check_solved", { method: "POST" })
              .then((response) => response.json())
              .then((data) => {
                  if (!data.is_solved) {
                      fetch("/next_step", { method: "POST" })
                          .then((response) => response.json())
                          .then((data) => {
                              if (data.next_move) {
                                  updateButtonColors(data.q_values);
                              }
                          });
                  } else {
                      resetButtonColors();
                      solutionPre.textContent = "魔方已经还原";
                  }
              });
      } else {
          fetch("/check_solved_one_face", { method: "POST" })
              .then((response) => response.json())
              .then((data) => {
                  if (!data.is_face_solved) {
                      fetch("/next_step_one_face", { method: "POST" })
                          .then((response) => response.json())
                          .then((data) => {
                              if (data.next_move) {
                                  updateButtonColors(data.q_values);
                              }
                          });
                  } else {
                      resetButtonColors();
                      solutionPre.textContent = "已有一面还原";
                  }
              });
      }
  });

  modeBtn.addEventListener("click", function () {
      resetButtonColors();
      if (!cancelSolution()) {
          return;
      }

      if (this.value === "full") {
          this.value = "one-face";
          solutionPre.textContent = "已切换到单面模式";
      } else {
          this.value = "full";
          solutionPre.textContent = "已切换到完整模式";
      }
      resetButtonColors();
  });
});

const keyToMove = {
  f: "F",
  r: "R",
  u: "U",
  g: "F'",
  t: "R'",
  i: "U'",
  z: "scramble",
  x: "solve",
  c: "next-step",
  j: "reset",
  k: "switch-mode"
};

document.addEventListener("keydown", function (event) {
  if (!cancelSolution()) {
      return;
  }

  const action = keyToMove[event.key];
  if (action) {
      if (action === "reset") {
          document.getElementById("reset-btn").click();
      } else if (action === "scramble") {
          document.getElementById("scramble-btn").click();
      } else if (action === "solve") {
          document.getElementById("solve-btn").click();
      } else if (action === "next-step") {
          document.getElementById("next-step-btn").click();
      } else if (action === "switch-mode") {
          document.getElementById("next-step-mode").click();
      } else {
          rotateCube(action);
      }
  }
});

const buttons = [
  document.getElementById("f-btn"),
  document.getElementById("r-btn"),
  document.getElementById("u-btn"),
  document.getElementById("f-prime-btn"),
  document.getElementById("r-prime-btn"),
  document.getElementById("u-prime-btn")
];

function resetButtonColors() {
  buttons.forEach(button => {
      button.style.backgroundColor = '';
  });
}

function rotateCube(move) {
  const solutionPre = document.getElementById("solution");
  fetch("/rotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move: move }),
  })
      .then((response) => response.json())
      .then((data) => {
          drawCube(data.cube_reduced);
          if (solutionInterval === null) {
              solutionPre.textContent = `${move}`;
          }
      });
  resetButtonColors();
}

let solutionInterval = null;

function animateSolution(solution) {
  let index = 0;
  const solutionPre = document.getElementById("solution");

  if (solutionInterval !== null) {
      clearInterval(solutionInterval);
  }

  solutionInterval = setInterval(() => {
      if (index < solution.length) {
          fetch("/rotate", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ move: solution[index] }),
          })
              .then((response) => response.json())
              .then((data) => {
                  drawCube(data.cube_reduced);
              });
          index++;
      } else {
          clearInterval(solutionInterval);
          solutionInterval = null;
          solutionPre.textContent = `解法: ${solution.join(" ")}`;
      }
  }, 120);
}

function simulateKeyPress(key) {
  const event = new KeyboardEvent("keydown", {
      key: key,
      bubbles: true,
  });
  document.dispatchEvent(event);
}

function updateButtonColors(q_values) {
  const avgQ = q_values.reduce((a, b) => a + b, 0) / q_values.length;
  const maxDiff = Math.max(...q_values.map(q => Math.abs(q - avgQ)));
  
  buttons.forEach((button, index) => {
      const diff = q_values[index] - avgQ;
      const normalizedDiff = Math.abs(diff) / maxDiff;
      const alpha = 0.2 + normalizedDiff * 0.8;
      const color = diff >= 0 ? 
          `rgba(0, 155, 72, ${alpha})` :
          `rgba(255, 40, 40, ${alpha})`;
      
      button.style.backgroundColor = color;
  });
}

function drawHeatmap(solutionProcess) {
  const modal = document.getElementById('heatmap-modal');
  const canvas = document.getElementById('heatmap-canvas');
  const span = document.getElementsByClassName("close")[0];
  const cellWidth = 80;
  const cellHeight = 40;
  const headerHeight = 40;
  const stepLabelWidth = 60;
  
  const dpr = window.devicePixelRatio || 1;
  
  canvas.width = (stepLabelWidth + cellWidth * 6) * dpr;
  canvas.height = (headerHeight + cellHeight * solutionProcess.length) * dpr;
  
  canvas.style.width = `${stepLabelWidth + cellWidth * 6}px`;
  canvas.style.height = `${headerHeight + cellHeight * solutionProcess.length}px`;
  
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  
  function drawBackground() {
    const actions = ['F', 'R', 'U', "F'", "R'", "U'"];
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, headerHeight);
    ctx.fillStyle = 'black';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('步骤', stepLabelWidth / 2, headerHeight / 2);
    
    actions.forEach((action, index) => {
      ctx.fillText(action, stepLabelWidth + index * cellWidth + cellWidth / 2, headerHeight / 2);
    });
    
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= solutionProcess.length; i++) {
      ctx.beginPath();
      ctx.moveTo(0, headerHeight + i * cellHeight);
      ctx.lineTo(canvas.width, headerHeight + i * cellHeight);
      ctx.stroke();
    }
    
    for (let i = 0; i <= 6; i++) {
      ctx.beginPath();
      ctx.moveTo(stepLabelWidth + i * cellWidth, 0);
      ctx.lineTo(stepLabelWidth + i * cellWidth, canvas.height);
      ctx.stroke();
    }
  }

  function drawStep(step, stepIndex, opacity) {
    ctx.fillStyle = `rgba(0, 0, 0, ${opacity})`;
    ctx.textAlign = 'center';
    ctx.fillText(`步骤 ${step.step + 1}`, stepLabelWidth / 2, headerHeight + stepIndex * cellHeight + cellHeight / 2);
    
    const stepAvgQ = step.q_values.reduce((a, b) => a + b, 0) / step.q_values.length;
    
    step.q_values.forEach((q, actionIndex) => {
      const x = stepLabelWidth + actionIndex * cellWidth;
      const y = headerHeight + stepIndex * cellHeight;
      
      const diff = q - stepAvgQ;
      const maxDiff = Math.max(...step.q_values.map(q => Math.abs(q - stepAvgQ))); 
      const normalizedDiff = maxDiff === 0 ? 0 : Math.abs(diff) / maxDiff;
      const alpha = (0.2 + normalizedDiff * 0.8) * opacity;
      
      const color = diff >= 0 ? 
        `rgba(0, 155, 72, ${alpha})` :
        `rgba(255, 40, 40, ${alpha})`;
      
      ctx.fillStyle = color;
      ctx.fillRect(x, y, cellWidth, cellHeight);
      
      if (step.action === actionIndex) {
        ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      }
    });
  }

  function animate(currentStep = 0) {
    if (currentStep >= solutionProcess.length) return;
    let opacity = 0;
    const fadeInterval = setInterval(() => {
      opacity += 0.1;
      drawBackground();
      for (let i = 0; i < currentStep; i++) {
        drawStep(solutionProcess[i], i, 1);
      }
      drawStep(solutionProcess[currentStep], currentStep, opacity);
      if (opacity >= 1) {
        clearInterval(fadeInterval);
        setTimeout(() => animate(currentStep + 1), 100);
      }
    }, 20);
  }

  const escHandler = function(event) {
    if (event.key === "Escape") {
      modal.style.display = "none";
      document.removeEventListener("keydown", escHandler);
    }
  };
  document.addEventListener("keydown", escHandler);

  modal.style.display = "block";
  drawBackground();
  animate();
  
  span.onclick = function () {
      modal.style.display = "none";
      document.removeEventListener("keydown", escHandler);
  }
  
  window.onclick = function (event) {
      if (event.target == modal) {
          modal.style.display = "none";
          document.removeEventListener("keydown", escHandler);
      }
  }
}