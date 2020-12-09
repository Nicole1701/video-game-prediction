// Render & Populate Table Head
function render_thead(data){
    d3.select("table")
        .append("thead")
    var keys = Object.keys(data[0].vg_data[0])
    keys.forEach(key=>
        d3.select("thead")
            .append("th")
            .text(key)
        )
}

// Render & Populate Table Body
function render_tbody(data){
    d3.select("table")
        .append("tbody")
    var keys = Object.keys(data[0].vg_data[0])
    var vg_data = data[0].vg_data
    console.log(vg_data.length)
    for(var i=0,vg_length=vg_data.length;i<vg_length;i++){
        var row_index = "row"+String(i)
        console.log(row_index)
        d3.select("tbody")
            .append("tr")
            .attr("id",row_index)
        for(var j=0,key_length=keys.length;j<key_length;j++){
            var row_id = "#"+String(row_index)
            var tdtext = vg_data[i][keys[j]]
            if(keys[j].includes("Sales") || keys[j].includes("Price")){
                if(keys[j]==="Price"){
                    tdtext = tdtext.toLocaleString('en-US',{style:'currency',currency:'USD'})
                }
                else{
                    tdtext = tdtext.toLocaleString()
                }
                d3.select(row_id)
                    .append("td")
                    .text(tdtext)
            }
            else{
                d3.select(row_id)
                .append("td")
                .text(tdtext)
            }
        }
    }
}

// TEST FILTER BODY FUNCTION
function testerz(data,platform,year,genre){
    d3.select("table")
        .append("tbody")
    var keys = Object.keys(data[0].vg_data[0])
    var vg_data = data[0].vg_data
    console.log(vg_data.length)
    for(var i=0,vg_length=vg_data.length;i<vg_length;i++){
        if(
            platform.includes(vg_data[i].Platform) &&
            year.includes(String(vg_data[i].Year)) &&
            genre.includes(vg_data[i].Genre)
        ){
            var row_index = "row"+String(i)
            console.log(row_index)
            d3.select("tbody")
                .append("tr")
                .attr("id",row_index)
            for(var j=0,key_length=keys.length;j<key_length;j++){
                var row_id = "#"+String(row_index)
                var tdtext = vg_data[i][keys[j]]
                if(keys[j].includes("Sales") || keys[j].includes("Price")){
                    if(keys[j]==="Price"){
                        tdtext = tdtext.toLocaleString('en-US',{style:'currency',currency:'USD'})
                    }
                    else{
                        tdtext = tdtext.toLocaleString()
                    }
                    d3.select(row_id)
                        .append("td")
                        .text(tdtext)
                }
                else{
                    d3.select(row_id)
                    .append("td")
                    .text(tdtext)
                }
            }
        }

    }
}

// Function for on page load with unfiltered data
function render_table(data){
    render_thead(data)
    var platform = data[0].platforms
    var year = data[0].years
    var genre = data[0].genres
    testerz(data,platform,year,genre)
}

// Outer function to render table on page load
d3.json("vg_data").then(data=>
    render_table(data)
    )

// Filtered Table Functions