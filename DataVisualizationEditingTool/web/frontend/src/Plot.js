import React, { useEffect, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import './Plot.css';

const Plot = ({ drawMode, setData }) => {
  const svgRef = useRef();
  const dataRef = useRef(null);

  useEffect(() => {
    axios.get('/api/data')
      .then(response => {
        dataRef.current = response.data;
        setData(response.data);
        render();
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, [setData]);

  const render = () => {
    if (dataRef.current) {
      const data = dataRef.current;
      const svg = d3.select(svgRef.current);
      const width = svg.node().getBoundingClientRect().width;
      const height = svg.node().getBoundingClientRect().height;
      const margin = { top: 20, right: 20, bottom: 30, left: 40 };

      const xScale = d3.scaleLinear()
        .domain(d3.extent(data.nodes, d => d[1]))
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain(d3.extent(data.nodes, d => d[2]))
        .range([height - margin.bottom, margin.top]);

      svg.selectAll('*').remove();

      svg.append('g').selectAll('.edge')
        .data(data.edges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('x1', d => xScale(data.nodes.find(n => n[0] === d[0])[1]))
        .attr('y1', d => yScale(data.nodes.find(n => n[0] === d[0])[2]))
        .attr('x2', d => xScale(data.nodes.find(n => n[0] === d[1])[1]))
        .attr('y2', d => yScale(data.nodes.find(n => n[0] === d[1])[2]))
        .attr('stroke', 'black')
        .attr('stroke-width', 0.5);

      const color = d3.scaleOrdinal(d3.schemeCategory10);
      svg.append('g').selectAll('.node')
        .data(data.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', d => xScale(d[1]))
        .attr('cy', d => yScale(d[2]))
        .attr('r', 5)
        .attr('fill', d => color(d[4]));

      svg.on('click', (event) => {
        if (!drawMode) return;

        const [x, y] = d3.pointer(event);
        const invertedX = xScale.invert(x);
        const invertedY = yScale.invert(y);

        const newNodeId = data.nodes.length > 0 ? Math.max(...data.nodes.map(n => n[0])) + 1 : 0;

        const newNode = [
          newNodeId,
          invertedX,
          invertedY,
          0,
          0
        ];

        data.nodes.push(newNode);
        setData({ ...data, nodes: data.nodes });
        render();
      });
    }
  };

  return (
    <div className="plot">
      <svg ref={svgRef} width="100%" height="100%"></svg>
    </div>
  );
};

export default Plot;
