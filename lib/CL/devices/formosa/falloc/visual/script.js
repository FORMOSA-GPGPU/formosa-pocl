const request = new XMLHttpRequest();
const file = prompt("Enter log file path:", "log.json")
request.open("GET", file, false);
request.send(null);
const jsonData = JSON.parse(request.responseText);
logs = jsonData.slice(1)

// Config memory size
const mem_start = jsonData[0]["start"]
const mem_size = jsonData[0]["size"]
document.getElementById("log_info").innerHTML = `File: ${file}<br/>Start address: ${mem_start} size : ${mem_size}`

// Returns a membar
function constructMemBar(list) {
    const membar = document.createElement("div");
    membar.className = "memory-bar";
    list.forEach(block => {
        const block_start = block["start_addr"];
        const block_size = block["size"];
        const pos_start = (block_start - mem_start) / mem_size * 100;
        const width = block_size / mem_size * 100;
        if (pos_start < 0 || pos_start + width > 103) {
            window.alert("Log file error: memory");
            throw new Error("log file error: memory");
        }
        const seg = document.createElement("div");

        seg.className = "mem-seg";
        seg.style = `left: ${pos_start}%; width: ${width}%;`;
        seg.setAttribute("data-memtip", `start: ${block_start} size: ${block_size}`);

        membar.appendChild(seg);
    })
    return membar
}

const logContainer = document.getElementById("logs");
/* Create block for each log */
logs.forEach(function (log) {
    const logBlock = document.createElement("div");
    logBlock.className = "log-block";
    /* Create log info */
    const logInfo = document.createElement("div");
    logInfo.className = "log-info";
    logInfo.textContent = `${log.operation} ${log.value}`
    logBlock.appendChild(logInfo)

    const bars = document.createElement("div");
    bars.className = "memory-bars";
    const free_list = log["free_list"];
    const alloc_list = log["allocated_list"];
    bars.appendChild(constructMemBar(free_list));
    bars.appendChild(constructMemBar(alloc_list));

    logBlock.appendChild(bars);
    logContainer.appendChild(logBlock);
});

document.querySelectorAll(".mem-seg").forEach(seg => {
    const memtip = document.createElement("div");
    document.body.appendChild(memtip);

    memtip.classList.add("memtip");
    memtip.textContent = seg.getAttribute("data-memtip");

    seg.addEventListener('mouseenter', function () {
        // Show the tooltip
        memtip.style.display = 'block';

        // Position the tooltip at the cursor
        const offsetX = 10; // Offset to avoid the tooltip directly on the cursor
        const offsetY = 40; // Vertical offset to avoid overlap with cursor


        document.addEventListener('mousemove', function (e) {
            memtip.style.left = `${e.clientX + offsetX}px`;  // Position horizontally
            memtip.style.top = `${e.clientY - offsetY}px`;  // Position vertically
        });
    });

    seg.addEventListener('mouseleave', function () {
        memtip.style.display = 'none';  // Hide the tooltip
    });
})


// Show the button when the user scrolls down 100px from the top
window.onscroll = function () {
    const btn = document.getElementById("scrollToTopBtn");
    if(btn == null)
        return;
    if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
        btn.style.display = "block";
    } else {
        btn.style.display = "none";
    }
};

// Function to scroll to the top
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

var xmas = false;

function toggleXMAS() {
    xmas = !xmas;
    const santa = document.getElementById("santa");
    const led = document.getElementById("led");
    if(xmas == true) {
        santa.style.visibility = "visible";
        led.style.visibility = "visible";
    } else {
        santa.style.visibility = "hidden";
        led.style.visibility = "hidden";
    }
    // Toggle color
    const darkgreen = "#299100";
    const lightgreen = "#4ee80d";
    const red = "#ff0000";
    const normalgreen = "#34c000";
    const white = "#ffffff";
    const black = "#000000";
    const blue = " #0a93ba";

    if(xmas == true) {
        document.body.style.backgroundColor = darkgreen;
        document.querySelectorAll(".mem-seg").forEach(seg => {
            seg.style.backgroundColor = red;
        })
        document.querySelectorAll(".log-block").forEach(block => {
            block.style.backgroundColor = lightgreen;
        })
        document.querySelectorAll(".memory-bar").forEach(bar => {
            bar.style.backgroundColor = "#ccc";
        })
    } else {
        document.body.style.backgroundColor = white;
        document.querySelectorAll(".mem-seg").forEach(seg => {
            seg.style.backgroundColor = normalgreen;
        })
        document.querySelectorAll(".log-block").forEach(block => {
            block.style.backgroundColor = white;
        })
        document.querySelectorAll(".memory-bar").forEach(bar => {
            bar.style.backgroundColor = black;
        })
    }
}